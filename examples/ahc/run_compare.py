import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env import CPUS_PER_TASK, discover_ahc


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


def _score_series(rows: list[dict]) -> tuple[list[int], list[float], list[float]]:
    steps: list[int] = []
    raw_scores: list[float] = []
    best_scores: list[float] = []
    running_best = -math.inf
    for idx, row in enumerate(rows):
        score = row.get("env/all/raw_score/max")
        if score is None:
            score = row.get("env/all/raw_score")
        if score is None:
            score = row.get("env/all/reward/max", 0.0)
        score = float(score)
        running_best = max(running_best, score)
        steps.append(idx)
        raw_scores.append(score)
        best_scores.append(running_best)
    return steps, raw_scores, best_scores


def _extract_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    last = rows[-1]
    scores = [float(r.get("env/all/raw_score/max", r.get("env/all/raw_score", 0.0))) for r in rows]
    return {
        "steps": len(rows),
        "final_score": scores[-1] if scores else float("nan"),
        "best_score": max(scores) if scores else float("nan"),
        "final_total_time_s": float(last.get("time/total", float("nan"))),
    }


def _print_summary(name: str, summary: dict, log_dir: Path) -> None:
    print(f"\n[{name}]")
    print(f"log_dir={log_dir}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")


def _plot_results(output_path: Path, runs: dict[str, list[dict]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, rows in runs.items():
        steps, raw_scores, best_scores = _score_series(rows)
        axes[0].plot(steps, raw_scores, marker="o", label=name)
        axes[1].plot(steps, best_scores, marker="o", label=name)

    axes[0].set_title("Raw Score By Step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Raw Score")
    axes[1].set_title("Best Score So Far")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Best Raw Score")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PUCT, HTA(h=1), and MAP-Elites on an AHC task and plot scores.")
    parser.add_argument("--problem-type", default="ahc039", choices=["ahc039", "ahc058"])
    parser.add_argument("--backend-type", default="local_inference", choices=["local_inference", "gemini_inference", "tinker_train"])
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--tokenizer-model-name", default=None)
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--renderer-name", default=None)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=CPUS_PER_TASK)
    parser.add_argument("--local-max-new-tokens", type=int, default=2048)
    parser.add_argument("--base-log-dir", default="tinker_log")
    parser.add_argument("--plot-path", default=None)
    args = parser.parse_args()

    common_kwargs = dict(
        problem_type=args.problem_type,
        backend_type=args.backend_type,
        model_name=args.model_name,
        tokenizer_model_name=args.tokenizer_model_name,
        local_model_path=args.local_model_path,
        renderer_name=args.renderer_name,
        num_steps=args.num_steps,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        num_cpus_per_task=args.num_cpus_per_task,
        local_max_new_tokens=args.local_max_new_tokens,
    )

    run_specs = [
        ("puct", {"sampler_type": "puct", "hta_commit_horizon": 1}),
        ("hta_h1", {"sampler_type": "hta", "hta_commit_horizon": 1}),
        ("map_elites", {"sampler_type": "map_elites_islands", "hta_commit_horizon": 1}),
    ]

    for run_name, overrides in run_specs:
        discover_ahc(**common_kwargs, **overrides)

    base_log_dir = Path(args.base_log_dir)
    log_dirs = {
        "puct": base_log_dir / f"{args.problem_type}-puct-{args.backend_type}",
        "hta_h1": base_log_dir / f"{args.problem_type}-hta-{args.backend_type}",
        "map_elites": base_log_dir / f"{args.problem_type}-map_elites_islands-{args.backend_type}",
    }

    rows_by_name = {name: _load_metrics(path) for name, path in log_dirs.items()}
    summaries = {name: _extract_summary(rows) for name, rows in rows_by_name.items()}
    for name, summary in summaries.items():
        _print_summary(name, summary, log_dirs[name])

    plot_path = (
        Path(args.plot_path)
        if args.plot_path is not None
        else base_log_dir / f"{args.problem_type}-compare-{args.backend_type}.png"
    )
    _plot_results(plot_path, rows_by_name)
    print(f"\n[plot]\npath={plot_path}")


if __name__ == "__main__":
    main()
