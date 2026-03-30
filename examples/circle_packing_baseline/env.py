import inspect
import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt_discover import DiscoverConfig, Environment, SandboxRewardEvaluator, State, discover


def validate_packing(centers, radii):
    n = centers.shape[0]

    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0 or np.isnan(radii[i]):
            return False

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False

    return True


def _baseline_code(centers: list[list[float]], radii: list[float]) -> str:
    centers_rows = ",\n        ".join(str([float(x), float(y)]) for x, y in centers)
    radii_row = ", ".join(str(float(r)) for r in radii)
    return f"""```python
import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    centers = np.array([
        {centers_rows}
    ], dtype=float)
    radii = np.array([{radii_row}], dtype=float)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
```"""


def _packing_signature(centers: list[list[float]], radii: list[float], baseline_name: str | None = None) -> list[float | str]:
    signature: list[float | str] = []
    if baseline_name is not None:
        signature.append(baseline_name)
    for x, y in centers:
        signature.extend([float(x), float(y)])
    signature.extend(float(r) for r in radii)
    return signature


def packing_behavior_descriptor_from_state(state: State) -> list[float] | None:
    construction = getattr(state, "construction", None)
    if not isinstance(construction, list):
        return None
    values = construction[:]
    if values and isinstance(values[0], str):
        values = values[1:]
    numeric = [float(x) for x in values if isinstance(x, (int, float))]
    if len(numeric) < 18:
        return None

    xs = np.array(numeric[0:12:2], dtype=float)
    ys = np.array(numeric[1:12:2], dtype=float)
    radii = np.array(numeric[12:18], dtype=float)
    centers = np.stack([xs, ys], axis=1)

    pairwise_gaps: list[float] = []
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            pairwise_gaps.append(dist - float(radii[i] + radii[j]))
    min_gap = min(pairwise_gaps) if pairwise_gaps else 0.0

    wall_clearances: list[float] = []
    for (x, y), r in zip(centers, radii):
        wall_clearances.extend([x - r, y - r, 1.0 - (x + r), 1.0 - (y + r)])
    mean_wall_clearance = float(np.mean(wall_clearances)) if wall_clearances else 0.0

    descriptor = [
        float(np.clip(np.mean(radii), 0.0, 1.0)),
        float(np.clip(np.std(radii), 0.0, 1.0)),
        float(np.clip(0.5 + 0.5 * np.tanh(4.0 * min_gap), 0.0, 1.0)),
        float(np.clip(0.5 + 0.5 * np.tanh(4.0 * mean_wall_clearance), 0.0, 1.0)),
    ]
    return descriptor


def _state_baseline_name(state: State, default: str) -> str:
    construction = getattr(state, "construction", None)
    if isinstance(construction, dict):
        baseline_name = construction.get("baseline")
        if isinstance(baseline_name, str) and baseline_name in BASELINE_LIBRARY:
            return baseline_name
    if isinstance(construction, list) and construction:
        first = construction[0]
        if isinstance(first, str) and first in BASELINE_LIBRARY:
            return first
    return default

BASELINE_LIBRARY: dict[str, dict] = {
    "strong_grid": {
        "name": "strong_grid",
        "description": "A strong equal-radius 3x2 grid baseline.",
        "centers": [
            [1.0 / 6.0, 1.0 / 6.0],
            [3.0 / 6.0, 1.0 / 6.0],
            [5.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 3.0 / 6.0],
            [3.0 / 6.0, 3.0 / 6.0],
            [5.0 / 6.0, 3.0 / 6.0],
        ],
        "radii": [1.0 / 6.0] * 6,
    },
    "corners_edges": {
        "name": "corners_edges",
        "description": "A conservative baseline using corners and edge centers.",
        "centers": [
            [0.15, 0.15],
            [0.50, 0.15],
            [0.85, 0.15],
            [0.15, 0.85],
            [0.50, 0.85],
            [0.85, 0.85],
        ],
        "radii": [0.10] * 6,
    },
    "staggered_rows": {
        "name": "staggered_rows",
        "description": "A weaker but valid staggered two-row baseline.",
        "centers": [
            [0.18, 0.22],
            [0.50, 0.22],
            [0.82, 0.22],
            [0.30, 0.58],
            [0.62, 0.58],
            [0.86, 0.58],
        ],
        "radii": [0.09] * 6,
    },
    "weak_grid": {
        "name": "weak_grid",
        "description": "A deliberately weak but valid grid baseline.",
        "centers": [
            [0.18, 0.18],
            [0.50, 0.18],
            [0.82, 0.18],
            [0.18, 0.50],
            [0.50, 0.50],
            [0.82, 0.50],
        ],
        "radii": [0.08] * 6,
    },
}


for baseline in BASELINE_LIBRARY.values():
    baseline["sum"] = float(sum(baseline["radii"]))
    baseline["code"] = _baseline_code(baseline["centers"], baseline["radii"])


def _make_baseline_state(baseline_name: str) -> State:
    baseline = BASELINE_LIBRARY[baseline_name]
    return State(
        timestep=-1,
        construction=_packing_signature(baseline["centers"], baseline["radii"], baseline_name),
        code=baseline["code"],
        value=baseline["sum"],
        observation=(
            f"Valid baseline '{baseline_name}' loaded with sum of radii {baseline['sum']:.6f}. "
            f"{baseline['description']}"
        ),
    )

class CirclePackingBaselineReward(SandboxRewardEvaluator):
    def get_program_entrypoint(self) -> str:
        return "run_packing"

    def preprocess_generation(self, generation: str, state: State) -> str:
        helper_src = inspect.getsource(validate_packing)
        return f"import numpy as np\n\n{helper_src}\n\n{generation}"

    def get_reward(self, code: str, state: State) -> float:
        output, error_msg = self.execute_code(code, state)
        if error_msg:
            return self._get_failure_entry(error_msg)

        centers, radii, _ = output
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        if centers.shape != (6, 2) or radii.shape != (6,):
            return self._get_failure_entry("Wrong output shapes; expected centers (6,2) and radii (6,).")

        if not validate_packing(centers, radii):
            return self._get_failure_entry("Packing is not valid.")

        sum_of_radii = float(np.sum(radii))
        return {
            "reward": sum_of_radii,
            "correctness": 1.0,
            "raw_score": sum_of_radii,
            "msg": f"Success; raw_score={sum_of_radii}",
            "result_construction": _packing_signature(centers.tolist(), radii.tolist()),
            "stdout": getattr(self, "_last_stdout", ""),
        }


class CirclePackingBaselineEnv(Environment):
    reward_function = CirclePackingBaselineReward
    state_type = State
    baseline_name = "strong_grid"

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        if problem_type != "6":
            raise ValueError("CirclePackingBaselineEnv only supports problem_type='6'")
        return _make_baseline_state(cls.baseline_name)

    @staticmethod
    def behavior_descriptor(state: State) -> list[float] | None:
        return packing_behavior_descriptor_from_state(state)

    def get_question(self) -> str:
        validator_src = inspect.getsource(validate_packing)
        current_state = self.state
        baseline_name = _state_baseline_name(current_state, self.baseline_name)
        baseline = BASELINE_LIBRARY[baseline_name]
        state_ctx = current_state.to_prompt(max(1.05, baseline["sum"] + 0.05), metric_name="sum of radii")

        return f"""You are editing a working Python baseline for packing 6 circles in the unit square.

Your job is to improve the existing valid program slightly while keeping it runnable and valid.

We will run this validator at runtime:
```python
{validator_src}
```

{state_ctx}

Important:
- Start from the existing baseline code above. Make small safe edits.
- Only return Python code inside one ```python block.
- Define exactly one function: run_packing().
- Do not execute run_packing() at top level.
- Do not use scipy or cvxpy. Use only numpy and math.
- Return:
  centers: np.ndarray with shape (6, 2)
  radii: np.ndarray with shape (6,)
  sum_radii: float
- It is okay to keep the same packing if you are unsure, but the program must run.
- Prefer simple deterministic code over fancy optimization.
- The current baseline variant is "{baseline_name}".

Goal:
- Keep all circles inside [0,1] x [0,1]
- No overlaps
- Try to improve the baseline sum of radii of {baseline["sum"]:.6f}
"""


class CirclePackingWeakBaselineEnv(CirclePackingBaselineEnv):
    baseline_name = "weak_grid"


class CirclePackingMultiBaselineEnv(CirclePackingBaselineEnv):
    baseline_cycle = itertools.cycle(["strong_grid", "corners_edges", "staggered_rows"])

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        if problem_type != "6":
            raise ValueError("CirclePackingMultiBaselineEnv only supports problem_type='6'")
        return _make_baseline_state(next(cls.baseline_cycle))


def discover_circle_packing_baseline(
    *,
    backend_type: str = "local_inference",
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    local_model_path: str | None = None,
    renderer_name: str | None = "qwen3_instruct",
    sampler_type: str = "puct",
    num_steps: int = 10,
    group_size: int = 1,
    groups_per_batch: int = 2,
    num_cpus_per_task: int = 1,
    hta_commit_horizon: int = 1,
):
    config = DiscoverConfig(
        env_type=CirclePackingBaselineEnv,
        problem_type="6",
        num_cpus_per_task=num_cpus_per_task,
        eval_timeout=530,
        experiment_name=f"baseline-circle-packing-6-{sampler_type}-{backend_type}",
        wandb_project=None,
        backend_type=backend_type,
        model_name=model_name,
        local_model_path=local_model_path,
        renderer_name=renderer_name,
        sampler_type=sampler_type,
        hta_commit_horizon=hta_commit_horizon,
        num_epochs=num_steps,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
    )
    discover(config)


def discover_circle_packing_weak_baseline(
    *,
    backend_type: str = "local_inference",
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    local_model_path: str | None = None,
    renderer_name: str | None = "qwen3_instruct",
    sampler_type: str = "puct",
    num_steps: int = 10,
    group_size: int = 1,
    groups_per_batch: int = 2,
    num_cpus_per_task: int = 1,
    hta_commit_horizon: int = 1,
):
    config = DiscoverConfig(
        env_type=CirclePackingWeakBaselineEnv,
        problem_type="6",
        num_cpus_per_task=num_cpus_per_task,
        eval_timeout=530,
        experiment_name=f"weak-baseline-circle-packing-6-{sampler_type}-{backend_type}",
        wandb_project=None,
        backend_type=backend_type,
        model_name=model_name,
        local_model_path=local_model_path,
        renderer_name=renderer_name,
        sampler_type=sampler_type,
        hta_commit_horizon=hta_commit_horizon,
        num_epochs=num_steps,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
    )
    discover(config)


def discover_circle_packing_multi_baseline(
    *,
    backend_type: str = "local_inference",
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    local_model_path: str | None = None,
    renderer_name: str | None = "qwen3_instruct",
    sampler_type: str = "puct",
    num_steps: int = 10,
    group_size: int = 1,
    groups_per_batch: int = 3,
    num_cpus_per_task: int = 1,
    hta_commit_horizon: int = 1,
):
    config = DiscoverConfig(
        env_type=CirclePackingMultiBaselineEnv,
        problem_type="6",
        num_cpus_per_task=num_cpus_per_task,
        eval_timeout=530,
        experiment_name=f"multi-baseline-circle-packing-6-{sampler_type}-{backend_type}",
        wandb_project=None,
        backend_type=backend_type,
        model_name=model_name,
        local_model_path=local_model_path,
        renderer_name=renderer_name,
        sampler_type=sampler_type,
        hta_commit_horizon=hta_commit_horizon,
        num_epochs=num_steps,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
    )
    discover(config)


if __name__ == "__main__":
    discover_circle_packing_baseline()
