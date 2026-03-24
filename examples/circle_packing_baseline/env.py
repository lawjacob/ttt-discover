import inspect
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


BASELINE_CENTERS = [
    [1.0 / 6.0, 1.0 / 6.0],
    [3.0 / 6.0, 1.0 / 6.0],
    [5.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 3.0 / 6.0],
    [3.0 / 6.0, 3.0 / 6.0],
    [5.0 / 6.0, 3.0 / 6.0],
]
BASELINE_RADII = [1.0 / 6.0] * 6
BASELINE_SUM = float(sum(BASELINE_RADII))
BASELINE_CODE = """```python
import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    centers = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [3.0 / 6.0, 1.0 / 6.0],
        [5.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 3.0 / 6.0],
        [3.0 / 6.0, 3.0 / 6.0],
        [5.0 / 6.0, 3.0 / 6.0],
    ], dtype=float)
    radii = np.full(6, 1.0 / 6.0, dtype=float)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
```"""


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
            "result_construction": [],
            "stdout": getattr(self, "_last_stdout", ""),
        }


class CirclePackingBaselineEnv(Environment):
    reward_function = CirclePackingBaselineReward
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        if problem_type != "6":
            raise ValueError("CirclePackingBaselineEnv only supports problem_type='6'")
        return State(
            timestep=-1,
            construction={"centers": BASELINE_CENTERS, "radii": BASELINE_RADII},
            code=BASELINE_CODE,
            value=BASELINE_SUM,
            observation=f"Valid baseline loaded with sum of radii {BASELINE_SUM:.6f}.",
        )

    def get_question(self) -> str:
        validator_src = inspect.getsource(validate_packing)
        state_ctx = self.initial_state.to_prompt(1.05, metric_name="sum of radii")

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

Goal:
- Keep all circles inside [0,1] x [0,1]
- No overlaps
- Try to improve the baseline sum of radii of {BASELINE_SUM:.6f}
"""


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
        num_epochs=num_steps,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
    )
    discover(config)


if __name__ == "__main__":
    discover_circle_packing_baseline()
