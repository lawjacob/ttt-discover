import argparse
import asyncio
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env import discover_erdos_min_overlap_async


def _load_metrics(log_dir: Path) -> list[dict]:
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    rows: list[dict] = []
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_step(rows: list[dict], predicate) -> int | None:
    for i, row in enumerate(rows):
        if predicate(row):
            return i
    return None


def _extract_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    last = rows[-1]
    raw_scores = [float(r["env/all/raw_score/max"]) for r in rows if "env/all/raw_score/max" in r]
    rewards = [float(r["env/all/reward/mean"]) for r in rows if "env/all/reward/mean" in r]
    correctness = [float(r["env/all/correctness"]) for r in rows if "env/all/correctness" in r]
    initial = float(rows[0].get("env/all/initial_raw_score", float("nan")))
    first_valid = _first_step(rows, lambda r: float(r.get("env/all/correctness", 0.0)) > 0.0)
    first_better = _first_step(
        rows,
        lambda r: (
            "env/all/raw_score/max" in r
            and math.isfinite(initial)
            and float(r["env/all/raw_score/max"]) < initial
        ),
    )
    return {
        "steps": len(rows),
        "initial_raw_score": initial,
        "best_raw_score_min": min(raw_scores) if raw_scores else float("nan"),
        "final_raw_score_min": float(last.get("env/all/raw_score/max", float("nan"))),
        "best_reward_mean": max(rewards) if rewards else float("nan"),
        "valid_rate": sum(1 for c in correctness if c > 0.0) / len(correctness) if correctness else 0.0,
        "first_valid_step": first_valid if first_valid is not None else -1,
        "first_better_than_initial_step": first_better if first_better is not None else -1,
        "final_total_time_s": float(last.get("time/total", float("nan"))),
        "final_effective_niches": float(last.get("hta/effective_niches", float("nan"))),
        "final_archive_coverage": float(last.get("map_elites/archive_coverage", float("nan"))),
    }


def _print_summary(name: str, summary: dict, log_dir: Path) -> None:
    print(f"\n[{name}]")
    print(f"log_dir={log_dir}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")


def _write_summary(outputs_path: Path, summaries: dict[str, dict]) -> None:
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    with outputs_path.open("w") as f:
        json.dump(summaries, f, indent=2)

    csv_path = outputs_path.with_suffix(".csv")
    keys = ["sampler"] + sorted({k for summary in summaries.values() for k in summary.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for sampler, summary in summaries.items():
            row = {"sampler": sampler}
            row.update(summary)
            writer.writerow(row)


async def _run(args) -> None:
    common_kwargs = dict(
        backend_type=args.backend_type,
        model_name=args.model_name,
        tokenizer_model_name=args.tokenizer_model_name,
        local_model_path=args.local_model_path,
        renderer_name=args.renderer_name,
        num_steps=args.num_steps,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        num_cpus_per_task=args.num_cpus_per_task,
        map_elites_num_islands=args.map_elites_num_islands,
        map_elites_cells_per_dim=args.map_elites_cells_per_dim,
        map_elites_migration_interval=args.map_elites_migration_interval,
        map_elites_migration_top_k=args.map_elites_migration_top_k,
        wandb_project=None,
    )

    plan = [
        ("puct", 1, f"{args.experiment_prefix}-puct-{args.backend_type}"),
        ("map_elites_islands", 1, f"{args.experiment_prefix}-map-elites-{args.backend_type}"),
        ("hta", args.hta_commit_horizon, f"{args.experiment_prefix}-hta-h{args.hta_commit_horizon}-{args.backend_type}"),
    ]
    for sampler_type, commit_horizon, experiment_name in plan:
        await discover_erdos_min_overlap_async(
            sampler_type=sampler_type,
            hta_commit_horizon=commit_horizon,
            experiment_name=experiment_name,
            **common_kwargs,
        )

    base_log_dir = Path(args.base_log_dir)
    summaries: dict[str, dict] = {}
    for sampler_type, commit_horizon, experiment_name in plan:
        log_dir = base_log_dir / experiment_name
        summary = _extract_summary(_load_metrics(log_dir))
        label = sampler_type if sampler_type != "hta" else f"hta_h{commit_horizon}"
        summaries[label] = summary
        _print_summary(label, summary, log_dir)

    _write_summary(Path(args.output_json), summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PUCT vs MAP-Elites vs HTA on Erdos minimum overlap.")
    parser.add_argument("--backend-type", default="local_inference", choices=["local_inference", "gemini_inference", "tinker_train"])
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--tokenizer-model-name", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--renderer-name", default="qwen3_instruct")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--hta-commit-horizon", type=int, default=1)
    parser.add_argument("--map-elites-num-islands", type=int, default=4)
    parser.add_argument("--map-elites-cells-per-dim", type=int, default=4)
    parser.add_argument("--map-elites-migration-interval", type=int, default=5)
    parser.add_argument("--map-elites-migration-top-k", type=int, default=1)
    parser.add_argument("--experiment-prefix", default="erdos-min-overlap")
    parser.add_argument("--base-log-dir", default="tinker_log")
    parser.add_argument("--output-json", default="tinker_log/erdos-min-overlap-compare-summary.json")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
