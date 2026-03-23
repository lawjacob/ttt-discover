import argparse
import json
from pathlib import Path

from env import discover_circle_packing


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


def _extract_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    last = rows[-1]
    rewards = [float(r["env/all/reward/mean"]) for r in rows if "env/all/reward/mean" in r]
    reward_maxes = [float(r["env/all/reward/max"]) for r in rows if "env/all/reward/max" in r]
    effective_niches = [float(r["hta/effective_niches"]) for r in rows if "hta/effective_niches" in r]
    return {
        "steps": len(rows),
        "final_reward_mean": float(last.get("env/all/reward/mean", float("nan"))),
        "best_reward_mean": max(rewards) if rewards else float("nan"),
        "best_reward_max": max(reward_maxes) if reward_maxes else float("nan"),
        "final_sampling_mean_s": float(last.get("env/all/time/sampling_mean", float("nan"))),
        "final_total_time_s": float(last.get("time/total", float("nan"))),
        "final_alpha": float(last.get("hta/alpha", float("nan"))),
        "final_effective_niches": effective_niches[-1] if effective_niches else float("nan"),
    }


def _print_summary(name: str, summary: dict, log_dir: Path) -> None:
    print(f"\n[{name}]")
    print(f"log_dir={log_dir}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")


def _print_comparison(puct: dict, hta: dict) -> None:
    print("\n[comparison]")
    compare_keys = [
        "final_reward_mean",
        "best_reward_mean",
        "best_reward_max",
        "final_sampling_mean_s",
        "final_total_time_s",
    ]
    for key in compare_keys:
        puct_value = puct.get(key, float("nan"))
        hta_value = hta.get(key, float("nan"))
        delta = hta_value - puct_value
        print(f"{key}: puct={puct_value:.6f} hta={hta_value:.6f} delta_hta_minus_puct={delta:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PUCT vs HTA on circle packing with the same local backend.")
    parser.add_argument("--num-circles", default="26")
    parser.add_argument("--backend-type", default="local_inference", choices=["local_inference", "tinker_train"])
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--groups-per-batch", type=int, default=4)
    parser.add_argument("--base-log-dir", default="tinker_log")
    args = parser.parse_args()

    common_kwargs = dict(
        num_circles=args.num_circles,
        backend_type=args.backend_type,
        model_name=args.model_name,
        local_model_path=args.local_model_path,
        num_steps=args.num_steps,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
    )

    for sampler_type in ("puct", "hta"):
        discover_circle_packing(sampler_type=sampler_type, **common_kwargs)

    base_log_dir = Path(args.base_log_dir)
    puct_log_dir = base_log_dir / f"test-circle-packing-{args.num_circles}-puct-{args.backend_type}"
    hta_log_dir = base_log_dir / f"test-circle-packing-{args.num_circles}-hta-{args.backend_type}"

    puct_summary = _extract_summary(_load_metrics(puct_log_dir))
    hta_summary = _extract_summary(_load_metrics(hta_log_dir))

    _print_summary("puct", puct_summary, puct_log_dir)
    _print_summary("hta", hta_summary, hta_log_dir)
    _print_comparison(puct_summary, hta_summary)


if __name__ == "__main__":
    main()
