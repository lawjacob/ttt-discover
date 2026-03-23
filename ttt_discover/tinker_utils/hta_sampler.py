"""Hierarchical Trajectory Allocation (HTA) sampler.

This sampler approximates the HTA design described in ``hta.pdf``:
- states are mapped into a deterministic behavior space
- niches are stable anchor cells in that behavior space
- allocation is split across niches vs. within a niche
- both levels score expected improvement per unit depth
- the inter-niche fraction is adjusted by a diversity constraint

The implementation stays lightweight enough to fit the existing TTT-Discover
interfaces, but avoids the previous "hash bucket + heuristic score" approach.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import hashlib
import math
import os
import threading

import numpy as np

from ttt_discover.tinker_utils.state import State, state_from_dict
from ttt_discover.tinker_utils.best_sequence_utils import _file_lock, _atomic_write_json, _read_json_or_default
from ttt_discover.tinker_utils.sampler import StateSampler, create_initial_state


@dataclass
class NicheStats:
    best_reward: float = float("-inf")
    visits: int = 0
    progress: float = 0.0
    uncertainty: float = 1.0
    stagnation: int = 0
    live_lineages: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "best_reward": self.best_reward,
            "visits": self.visits,
            "progress": self.progress,
            "uncertainty": self.uncertainty,
            "stagnation": self.stagnation,
            "live_lineages": sorted(self.live_lineages),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NicheStats":
        return cls(
            best_reward=d.get("best_reward", float("-inf")),
            visits=int(d.get("visits", 0)),
            progress=float(d.get("progress", 0.0)),
            uncertainty=float(d.get("uncertainty", 1.0)),
            stagnation=int(d.get("stagnation", 0)),
            live_lineages=set(d.get("live_lineages", [])),
        )


@dataclass
class Lineage:
    id: str
    niche_id: str
    root_state_id: str
    frontier_state_id: str
    depth: int = 0
    recent_improvement: float = 0.0
    credit_score: float = 0.0
    stagnant_steps: int = 0
    parent_lineage_id: Optional[str] = None
    operator_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "niche_id": self.niche_id,
            "root_state_id": self.root_state_id,
            "frontier_state_id": self.frontier_state_id,
            "depth": self.depth,
            "recent_improvement": self.recent_improvement,
            "credit_score": self.credit_score,
            "stagnant_steps": self.stagnant_steps,
            "parent_lineage_id": self.parent_lineage_id,
            "operator_history": self.operator_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Lineage":
        return cls(
            id=d["id"],
            niche_id=d["niche_id"],
            root_state_id=d.get("root_state_id", d["frontier_state_id"]),
            frontier_state_id=d["frontier_state_id"],
            depth=int(d.get("depth", 0)),
            recent_improvement=float(d.get("recent_improvement", 0.0)),
            credit_score=float(d.get("credit_score", 0.0)),
            stagnant_steps=int(d.get("stagnant_steps", 0)),
            parent_lineage_id=d.get("parent_lineage_id"),
            operator_history=d.get("operator_history", []),
        )


class HTASampler(StateSampler):
    """Hierarchical Trajectory Allocation sampler.

    The sampler maintains persistent niches and active lineages. It outputs a
    batch of frontier states, allocating a fraction of the batch to inter-niche
    exploration and the remainder to intra-niche depth.
    """

    def __init__(
        self,
        file_path: str,
        env_type: type,
        problem_type: str = "",
        max_buffer_size: int = 2000,
        batch_size: int = 1,
        num_niches: int = 32,
        alpha_init: float = 0.5,
        alpha_step: float = 0.05,
        stagnation_window: int = 15,
        inter_fraction_floor: float = 0.2,
        inter_fraction_ceiling: float = 0.8,
        resume_step: int | None = None,
    ):
        if num_niches <= 0:
            raise ValueError("num_niches must be positive")
        self.file_path = file_path
        self.env_type = env_type
        self.problem_type = problem_type
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.num_niches = num_niches
        self.alpha = alpha_init
        self.alpha_step = alpha_step
        self.stagnation_window = stagnation_window
        self.inter_fraction_floor = inter_fraction_floor
        self.inter_fraction_ceiling = inter_fraction_ceiling

        self._states: list[State] = []
        self._state_by_id: dict[str, State] = {}
        self._niches: dict[str, NicheStats] = {}
        self._lineages: dict[str, Lineage] = {}
        self._state_to_lineage: dict[str, str] = {}
        self._T = 0
        self._lock = threading.Lock()
        self._anchor_dim = 8
        self._anchors = self._build_anchors()
        self._parent_credit: dict[str, float] = {}
        self._edge_credit: dict[str, float] = {}
        self._operator_credit: dict[str, float] = {}
        self._credit_decay = 0.97

        if resume_step is not None:
            self._load(resume_step)
        if not self._states:
            for _ in range(batch_size):
                state = create_initial_state(self.env_type, self.problem_type)
                self._register_state(state)
                self._ensure_lineage_for_state(state, parent_lineage_id=None)
            self._save(0)

    # ---------------------- core interface ----------------------
    def sample_states(self, num_states: int) -> list[State]:
        with self._lock:
            if not self._states:
                return []
            niche_distribution = self._niche_distribution()
            self._update_alpha(niche_distribution)

            num_inter = max(0, min(num_states, int(math.ceil(self.alpha * num_states))))
            num_intra = max(0, num_states - num_inter)

            picked: list[State] = []
            if num_inter > 0:
                picked.extend(self._pick_inter_niche(num_inter, niche_distribution))
            if num_intra > 0:
                picked.extend(self._pick_intra_niche(num_intra, niche_distribution))

            if len(picked) < num_states:
                # Fallback: fill with top states by value
                remaining = num_states - len(picked)
                picked.extend(self._fallback_states(remaining))

            return picked

    def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
        if not states:
            return
        assert len(states) == len(parent_states)
        with self._lock:
            for child, parent in zip(states, parent_states):
                self._register_state(child)
                self._update_lineage(child, parent)
            self._decay_stats()
            self._prune_stagnant_lineages()
            self._prune_buffer()
            if save:
                self._save(step if step is not None else self._T)

    def flush(self, step: int | None = None):
        with self._lock:
            self._decay_stats()
            self._prune_stagnant_lineages()
            self._prune_buffer()
            self._save(step if step is not None else self._T)

    def record_failed_rollout(self, parent: State):
        with self._lock:
            niche_id = self._assign_niche(parent)
            niche = self._niches.setdefault(niche_id, NicheStats())
            niche.visits += 1
            niche.stagnation += 1
            self._T += 1
            self._save(self._T)

    # ---------------------- internal helpers ----------------------
    def _assign_niche(self, state: State) -> str:
        vec = self._behavior_vector(state)
        distances = np.linalg.norm(self._anchors - vec[None, :], axis=1)
        bucket = int(np.argmin(distances))
        return f"niche_{bucket:03d}"

    def _register_state(self, state: State):
        self._states.append(state)
        self._state_by_id[state.id] = state
        niche_id = self._assign_niche(state)
        self._niches.setdefault(niche_id, NicheStats())

    def _ensure_lineage_for_state(self, state: State, parent_lineage_id: Optional[str]):
        niche_id = self._assign_niche(state)
        lineage_id = f"lin_{state.id}"
        lineage = Lineage(
            id=lineage_id,
            niche_id=niche_id,
            root_state_id=state.id,
            frontier_state_id=state.id,
            depth=max(0, state.timestep),
            parent_lineage_id=parent_lineage_id,
        )
        self._lineages[lineage_id] = lineage
        self._state_to_lineage[state.id] = lineage_id
        self._niches.setdefault(niche_id, NicheStats()).live_lineages.add(lineage_id)

    def _update_lineage(self, child: State, parent: State):
        child_niche = self._assign_niche(child)
        parent_lineage_id = self._state_to_lineage.get(parent.id)
        if parent_lineage_id is None:
            # Treat as a fresh lineage if parent not tracked (resume edge case)
            self._ensure_lineage_for_state(child, parent_lineage_id=None)
            return

        lineage = self._lineages[parent_lineage_id]
        lineage.frontier_state_id = child.id
        lineage.depth += 1
        improvement = 0.0
        if child.value is not None and parent.value is not None:
            improvement = float(child.value) - float(parent.value)
        lineage.recent_improvement = 0.7 * lineage.recent_improvement + 0.3 * improvement
        lineage.stagnant_steps = 0 if improvement > 0 else lineage.stagnant_steps + 1
        operator = self._operator_signature(parent, child)
        lineage.operator_history = (lineage.operator_history + [operator])[-8:]
        lineage.credit_score = 0.7 * lineage.credit_score + 0.3 * self._credit_signal(parent, child, operator, improvement)

        self._state_to_lineage[child.id] = lineage.id

        niche = self._niches.setdefault(child_niche, NicheStats())
        niche.visits += 1
        niche.best_reward = max(niche.best_reward, float(child.value) if child.value is not None else float("-inf"))
        niche.progress = 0.7 * niche.progress + 0.3 * max(0.0, improvement)
        niche.uncertainty = 0.8 * niche.uncertainty + 0.2 * abs(improvement)
        niche.stagnation = 0 if improvement > 0 else niche.stagnation + 1
        niche.live_lineages.add(lineage.id)

        self._T += 1
        self._update_credit(parent, child, operator, improvement)

    def _niche_distribution(self) -> dict[str, float]:
        visits = np.array([max(1, n.visits) for n in self._niches.values()], dtype=float)
        total = float(visits.sum())
        if total == 0:
            return {nid: 1.0 / max(1, len(self._niches)) for nid in self._niches}
        return {nid: v / total for nid, v in zip(self._niches.keys(), visits)}

    def _update_alpha(self, niche_dist: dict[str, float]):
        # Adaptive diversity constraint: increase inter-niche allocation when
        # the effective number of active niches falls below target coverage.
        p = np.array(list(niche_dist.values()), dtype=float)
        if p.size == 0:
            self.alpha = max(self.inter_fraction_floor, self.alpha)
            return
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        effective_niches = math.exp(entropy)
        target_effective_niches = max(2.0, min(self.num_niches, math.sqrt(max(1.0, self._T + 1.0)) + 1.0))
        diversity_gap = target_effective_niches - effective_niches

        stagnation = np.mean([n.stagnation for n in self._niches.values()]) if self._niches else 0.0
        expected_gain = np.mean(
            [self._lineage_expected_improvement_per_depth(lin) for lin in self._lineages.values()]
        ) if self._lineages else 0.0

        if diversity_gap > 0:
            self.alpha = min(
                self.inter_fraction_ceiling,
                self.alpha + self.alpha_step * (1.0 + diversity_gap / max(1.0, target_effective_niches)),
            )
        if stagnation > self.stagnation_window:
            self.alpha = min(self.inter_fraction_ceiling, self.alpha + self.alpha_step)
        if diversity_gap <= 0 and expected_gain > 0:
            self.alpha = max(self.inter_fraction_floor, self.alpha - 0.5 * self.alpha_step)
        self.alpha = float(np.clip(self.alpha, self.inter_fraction_floor, self.inter_fraction_ceiling))

    def _target_niche_distribution(self, niche_dist: dict[str, float]) -> dict[str, float]:
        if not self._niches:
            return {}
        scores: dict[str, float] = {}
        for nid, stats in self._niches.items():
            diversity_deficit = 1.0 - niche_dist.get(nid, 0.0)
            stagnation_score = min(1.0, stats.stagnation / max(1.0, self.stagnation_window))
            score = (
                0.25 * self._safe(stats.best_reward)
                + 0.20 * stats.progress
                + 0.20 * stats.uncertainty
                + 0.15 * stagnation_score
                + 0.20 * diversity_deficit
            )
            scores[nid] = score
        score_values = np.array(list(scores.values()), dtype=float)
        if score_values.size == 0:
            return {}
        score_values = score_values - np.max(score_values)
        probs = np.exp(score_values)
        probs = probs / probs.sum()
        return {nid: float(p) for nid, p in zip(scores.keys(), probs)}

    def _pick_inter_niche(self, num_states: int, niche_dist: dict[str, float]) -> list[State]:
        # Across niches: choose where expected improvement per unit depth is high,
        # while compensating under-covered niches to satisfy the diversity constraint.
        target_dist = self._target_niche_distribution(niche_dist)
        scores = sorted(target_dist.items(), key=lambda x: x[1], reverse=True)

        picked_states: list[State] = []
        for nid, _ in scores:
            if len(picked_states) >= num_states:
                break
            frontier = self._best_frontier_in_niche(nid)
            if frontier is not None:
                picked_states.append(frontier)
        return picked_states

    def _pick_intra_niche(self, num_states: int, niche_dist: dict[str, float]) -> list[State]:
        # Within a niche: deepen the lineages with the strongest expected
        # improvement per unit depth.
        target_dist = self._target_niche_distribution(niche_dist)
        niche_order = [nid for nid, _ in sorted(target_dist.items(), key=lambda x: x[1], reverse=True)]
        picked: list[State] = []
        used_lineages: set[str] = set()
        while len(picked) < num_states and niche_order:
            progress = False
            for nid in niche_order:
                candidates: list[tuple[float, Lineage]] = []
                for lid in self._niches.get(nid, NicheStats()).live_lineages:
                    if lid in used_lineages or lid not in self._lineages:
                        continue
                    lin = self._lineages[lid]
                    score = self._lineage_expected_improvement_per_depth(lin)
                    candidates.append((score, lin))
                candidates.sort(reverse=True, key=lambda x: x[0])
                if not candidates:
                    continue
                _, lin = candidates[0]
                st = self._state_by_id.get(lin.frontier_state_id)
                if st is not None:
                    picked.append(st)
                    used_lineages.add(lin.id)
                    progress = True
                    if len(picked) >= num_states:
                        break
            if not progress:
                break
        return picked

    def _best_frontier_in_niche(self, niche_id: str) -> Optional[State]:
        candidates = [self._state_by_id.get(self._lineages[lin_id].frontier_state_id) for lin_id in self._niches.get(niche_id, NicheStats()).live_lineages]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return None
        candidates.sort(key=lambda s: self._safe(s.value), reverse=True)
        return candidates[0]

    def _fallback_states(self, k: int) -> list[State]:
        if not self._states:
            return []
        by_value = sorted(self._states, key=lambda s: self._safe(s.value), reverse=True)
        return by_value[:k]

    def _behavior_vector(self, state: State) -> np.ndarray:
        code = getattr(state, "code", "") or ""
        observation = getattr(state, "observation", "") or ""
        construction = getattr(state, "construction", None)
        construction_len = len(construction) if construction else 0
        code_lines = code.count("\n") + (1 if code.strip() else 0)
        code_len = len(code)
        obs_len = len(observation)
        value = self._safe(getattr(state, "value", None))
        timestep = max(0, int(getattr(state, "timestep", 0)))

        h1 = self._hash_unit_interval(code or state.id, salt="code")
        h2 = self._hash_unit_interval(str(construction) if construction else state.id, salt="construction")
        h3 = self._hash_unit_interval(observation or state.id, salt="observation")

        vec = np.array(
            [
                math.tanh(value / 10.0) if math.isfinite(value) else -1.0,
                math.tanh(code_len / 2000.0),
                math.tanh(code_lines / 200.0),
                math.tanh(construction_len / 200.0),
                math.tanh(obs_len / 1000.0),
                math.tanh(timestep / 50.0),
                2.0 * h1 - 1.0,
                0.5 * (2.0 * h2 - 1.0) + 0.5 * (2.0 * h3 - 1.0),
            ],
            dtype=float,
        )
        norm = np.linalg.norm(vec)
        return vec if norm <= 0 else vec / norm

    def _build_anchors(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        anchors = rng.normal(size=(self.num_niches, self._anchor_dim))
        norms = np.linalg.norm(anchors, axis=1, keepdims=True)
        norms = np.where(norms <= 0, 1.0, norms)
        return anchors / norms

    @staticmethod
    def _hash_unit_interval(text: str, *, salt: str) -> float:
        digest = hashlib.sha1(f"{salt}:{text}".encode("utf-8", errors="ignore")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64 - 1)

    def _lineage_expected_improvement_per_depth(self, lineage: Lineage) -> float:
        stats = self._niches.get(lineage.niche_id, NicheStats())
        frontier = self._state_by_id.get(lineage.frontier_state_id)
        root = self._state_by_id.get(lineage.root_state_id)
        observed_gain = 0.0
        if frontier is not None and root is not None and frontier.value is not None and root.value is not None:
            observed_gain = max(0.0, float(frontier.value) - float(root.value))
        predicted_progress = 0.5 * max(0.0, lineage.recent_improvement) + 0.5 * max(0.0, stats.progress)
        expected_improvement = (
            0.35 * max(0.0, lineage.recent_improvement)
            + 0.25 * predicted_progress
            + 0.15 * max(0.0, stats.uncertainty)
            + 0.15 * max(0.0, lineage.credit_score)
            + 0.10 * observed_gain
        )
        depth_cost = max(1.0, math.sqrt(1.0 + lineage.depth))
        return expected_improvement / depth_cost

    def _operator_signature(self, parent: State, child: State) -> str:
        if (parent.code or "") != (child.code or ""):
            return "code_edit"
        if (parent.construction or []) != (child.construction or []):
            return "construction_edit"
        if (parent.observation or "") != (child.observation or ""):
            return "observation_shift"
        return "state_refine"

    def _credit_signal(self, parent: State, child: State, operator: str, improvement: float) -> float:
        parent_credit = self._parent_credit.get(parent.id, 0.0)
        edge_credit = self._edge_credit.get(f"{parent.id}->{child.id}", 0.0)
        operator_credit = self._operator_credit.get(operator, 0.0)
        return 0.5 * max(0.0, improvement) + 0.2 * parent_credit + 0.15 * edge_credit + 0.15 * operator_credit

    def _update_credit(self, parent: State, child: State, operator: str, improvement: float) -> None:
        signal = max(0.0, improvement)
        edge_key = f"{parent.id}->{child.id}"
        self._parent_credit[parent.id] = 0.8 * self._parent_credit.get(parent.id, 0.0) + 0.2 * signal
        self._edge_credit[edge_key] = 0.8 * self._edge_credit.get(edge_key, 0.0) + 0.2 * signal
        self._operator_credit[operator] = 0.8 * self._operator_credit.get(operator, 0.0) + 0.2 * signal

    def _decay_stats(self) -> None:
        for stats in self._niches.values():
            stats.progress *= self._credit_decay
            stats.uncertainty = max(0.05, stats.uncertainty * self._credit_decay)
        self._parent_credit = {k: v * self._credit_decay for k, v in self._parent_credit.items() if v * self._credit_decay > 1e-6}
        self._edge_credit = {k: v * self._credit_decay for k, v in self._edge_credit.items() if v * self._credit_decay > 1e-6}
        self._operator_credit = {k: v * self._credit_decay for k, v in self._operator_credit.items() if v * self._credit_decay > 1e-6}

    def _prune_stagnant_lineages(self) -> None:
        to_remove = {
            lid
            for lid, lin in self._lineages.items()
            if lin.stagnant_steps > self.stagnation_window and lin.credit_score <= 0 and lin.recent_improvement <= 0
        }
        if not to_remove:
            return
        self._lineages = {lid: lin for lid, lin in self._lineages.items() if lid not in to_remove}
        self._state_to_lineage = {sid: lid for sid, lid in self._state_to_lineage.items() if lid not in to_remove}
        for stats in self._niches.values():
            stats.live_lineages = {lid for lid in stats.live_lineages if lid not in to_remove}

    def _prune_buffer(self):
        if len(self._states) <= self.max_buffer_size:
            return
        quota = max(1, self.max_buffer_size // max(1, len(self._niches)))
        keep: list[State] = []
        for nid in sorted(self._niches.keys()):
            niche_states = [s for s in self._states if self._assign_niche(s) == nid]
            niche_states.sort(key=lambda s: self._safe(s.value), reverse=True)
            keep.extend(niche_states[:quota])
        if len(keep) < self.max_buffer_size:
            keep_ids = {s.id for s in keep}
            remainder = [s for s in sorted(self._states, key=lambda s: self._safe(s.value), reverse=True) if s.id not in keep_ids]
            keep.extend(remainder[: self.max_buffer_size - len(keep)])
        keep = keep[: self.max_buffer_size]
        keep_ids = {s.id for s in keep}
        self._states = keep
        self._state_by_id = {s.id: s for s in keep}
        self._state_to_lineage = {sid: lid for sid, lid in self._state_to_lineage.items() if sid in keep_ids}
        # Drop orphaned lineages
        live_lineages = {lid for lid in self._state_to_lineage.values()}
        self._lineages = {lid: lin for lid, lin in self._lineages.items() if lid in live_lineages}
        for nid, stats in self._niches.items():
            stats.live_lineages = {lid for lid in stats.live_lineages if lid in live_lineages}

    @staticmethod
    def _safe(x: Optional[float]) -> float:
        return float(x) if x is not None else float("-inf")

    # ---------------------- persistence ----------------------
    def _sampler_file_for_step(self, step: int) -> str:
        base_name = self.file_path.replace(".json", "")
        return f"{base_name}_step_{step:06d}.json"

    def _save(self, step: int | None):
        if step is None:
            step = self._T
        path = self._sampler_file_for_step(step)
        store = {
            "step": step,
            "states": [s.to_dict() for s in self._states],
            "niches": {nid: n.to_dict() for nid, n in self._niches.items()},
            "lineages": {lid: lin.to_dict() for lid, lin in self._lineages.items()},
            "state_to_lineage": self._state_to_lineage,
            "parent_credit": self._parent_credit,
            "edge_credit": self._edge_credit,
            "operator_credit": self._operator_credit,
            "alpha": self.alpha,
            "T": self._T,
            "num_niches": self.num_niches,
        }
        with _file_lock(f"{path}.lock"):
            _atomic_write_json(path, store)

    def _load(self, step: int):
        path = self._sampler_file_for_step(step)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot resume HTA sampler from {path}")
        with _file_lock(f"{path}.lock"):
            store = _read_json_or_default(path, default=None)
        if store is None:
            raise ValueError(f"Failed to load HTA sampler from {path}")

        self.num_niches = int(store.get("num_niches", self.num_niches))
        self.alpha = float(store.get("alpha", self.alpha))
        self._T = int(store.get("T", 0))

        self._states = [state_from_dict(s, state_type=self.env_type.state_type) for s in store.get("states", [])]
        self._state_by_id = {s.id: s for s in self._states}
        self._niches = {nid: NicheStats.from_dict(d) for nid, d in (store.get("niches", {}) or {}).items()}
        self._lineages = {lid: Lineage.from_dict(d) for lid, d in (store.get("lineages", {}) or {}).items()}
        self._state_to_lineage = {k: v for k, v in (store.get("state_to_lineage", {}) or {}).items() if k in self._state_by_id}
        self._parent_credit = {k: float(v) for k, v in (store.get("parent_credit", {}) or {}).items()}
        self._edge_credit = {k: float(v) for k, v in (store.get("edge_credit", {}) or {}).items()}
        self._operator_credit = {k: float(v) for k, v in (store.get("operator_credit", {}) or {}).items()}

        # Reattach live lineage sets after filtering
        live_lineages = {lid for lid in self._lineages.keys()}
        for stats in self._niches.values():
            stats.live_lineages = {lid for lid in stats.live_lineages if lid in live_lineages}

    # ---------------------- diagnostics ----------------------
    def get_sample_stats(self) -> dict:
        with self._lock:
            buffer_values = [s.value for s in self._states]
            def _stats(values, prefix: str):
                arr = np.array([v for v in values if v is not None])
                if len(arr) == 0:
                    return {}
                return {
                    f"{prefix}/mean": float(np.mean(arr)),
                    f"{prefix}/std": float(np.std(arr)),
                    f"{prefix}/min": float(np.min(arr)),
                    f"{prefix}/max": float(np.max(arr)),
                }
            stats = {
                "hta/buffer_size": len(self._states),
                "hta/num_niches": self.num_niches,
                "hta/alpha": self.alpha,
                "hta/T": self._T,
            }
            niche_dist = self._niche_distribution() if self._niches else {}
            if niche_dist:
                p = np.array(list(niche_dist.values()), dtype=float)
                stats["hta/effective_niches"] = float(math.exp(-(p * np.log(p + 1e-12)).sum()))
            stats.update(_stats(buffer_values, "hta/buffer_value"))
            return stats
