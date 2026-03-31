import asyncio
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt_discover import Environment, SandboxRewardEvaluator, State, DiscoverConfig, discover
from ttt_discover.discovery import discover_impl


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")
    
    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")
    
    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")
    
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")
    
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
    
    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)
    
    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
    
    dx = 2.0 / n_points
    
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)
    
    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")
    
    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    
    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def verify_erdos_solution(result: tuple[np.ndarray, float, int]) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


class ErdosMinOverlapRewardEvaluator(SandboxRewardEvaluator):
    def get_program_entrypoint(self) -> str:
        return "run"

    def preprocess_generation(self, generation, state) -> str:
        import inspect
        if self.verifier_src is None:
            return generation

        verifier_src = inspect.getsource(self.verifier_src)
        numpy_import = "import numpy as np"
        
        base = numpy_import + "\n\n" + verifier_src + "\n\n"
        
        # State with construction is required - no silent fallback
        if state is None:
            raise ValueError(
                "state is required for preprocess_generation. "
                "Use ExperienceSampler to provide initial state with construction."
            )
        if state.construction is not None:
            initial_h_values = f"initial_h_values = np.array({list(state.construction)!r})"
            base += initial_h_values + "\n\n"

        return base + generation

    def get_reward(self, code: str, state: State) -> float:
        output, error_msg = self.execute_code(code, state)
        if error_msg: 
            return self._get_failure_entry(error_msg)

        if not verify_erdos_solution(output):
            return self._get_failure_entry("Invalid solution.")
        h_values, c5_bound, n_points = output
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)

        return {
            "reward": float(1.0 / (1e-8 + c5_bound)),
            "correctness": 1.0,
            "raw_score": c5_bound,
            "msg": f"C5 bound: {c5_bound}",
            "result_construction": list(h_values),
            "stdout": getattr(self, '_last_stdout', ''),
        }


class ErdosMinOverlapEnv(Environment):
    reward_function = ErdosMinOverlapRewardEvaluator
    state_type = State
    max_construction_len = 1000

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        rng = np.random.default_rng()
        n_points = rng.integers(40, 100)
        construction = np.ones(n_points) * 0.5
        perturbation = rng.uniform(-0.4, 0.4, n_points)
        perturbation = perturbation - np.mean(perturbation)
        construction = construction + perturbation
        dx = 2.0 / n_points
        correlation = np.correlate(construction, 1 - construction, mode="full") * dx
        c5_bound = float(np.max(correlation))
        return State(timestep=-1, code="", value=-c5_bound, construction=list(construction))

    def is_maximize(self) -> bool:
        return False # Minimize upper bound

    @staticmethod
    def behavior_descriptor(state: State) -> list[float] | None:
        construction = getattr(state, "construction", None)
        if not isinstance(construction, list) or not construction:
            return None
        arr = np.array(construction, dtype=float)
        dx = 2.0 / len(arr)
        correlation = np.correlate(arr, 1.0 - arr, mode="full") * dx
        c5_bound = float(np.max(correlation))
        descriptor = [
            float(np.clip(np.mean(arr), 0.0, 1.0)),
            float(np.clip(np.std(arr), 0.0, 1.0)),
            float(np.clip(c5_bound / 1.0, 0.0, 1.0)),
            float(np.clip(len(arr) / 1000.0, 0.0, 1.0)),
        ]
        return descriptor

    def get_question(self) -> str:
        state = self.initial_state
        state_ctx = state.to_prompt(0.3808, metric_name="C₅ bound", maximize=False)
        
        # Construct construction section
        construction_section = ""
        if hasattr(state, 'construction') and state.construction is not None and len(state.construction) > 0:
            construction_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""

        # Construct code section
        if state.code and state.code.strip():
            code_section = '''Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.'''
        else:
            code_section = '''Write code to optimize this construction.'''

        # Construct final prompt
        return f'''You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: 1000s for your code to run
- **CPUs**: 2 available

## Rules
- Define `run(seed=42, budget_s=1000, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `initial_h_values` is a global numpy array already defined for you when an initial construction is available
- `evaluate_erdos_solution(h_values, c5_bound, n_points)` is a verifier that checks a completed candidate; it is not an optimizer and it requires exactly 3 arguments
- Do not import `initial_h_values` or `evaluate_erdos_solution` from any module; they already exist in the execution environment as globals
- Do not shadow those names by redefining them
- Your function must complete within budget_s seconds and return the best solution found

## Required Output Shape
- Return exactly one tuple: `(h_values, c5_bound, n_points)`
- `h_values` must be a 1D numpy array or a list convertible to a 1D numpy array
- `n_points` must equal `len(h_values)`
- `c5_bound` must equal `max(np.correlate(h_values, 1 - h_values, mode="full") * (2.0 / n_points))`
- Ensure `sum(h_values) == n_points / 2` up to numerical precision before reporting `c5_bound`

## Forbidden Patterns
- No example usage
- No top-level code that calls `run()`
- No `if __name__ == "__main__":`
- No plotting
- No printing required
- No fake imports such as `from evaluate_erdos_solution import ...` or `from initial_h_values import ...`
- Do not reference variables that are not defined inside your code or provided globals

## Minimal Valid Template
Use this exact interface shape and then improve the optimization logic inside it:

```python
import numpy as np
from scipy.optimize import minimize

def compute_c5(h_values):
    h_values = np.asarray(h_values, dtype=float)
    n_points = len(h_values)
    dx = 2.0 / n_points
    return float(np.max(np.correlate(h_values, 1.0 - h_values, mode="full") * dx))

def project_feasible(h_values):
    h_values = np.asarray(h_values, dtype=float)
    h_values = np.clip(h_values, 0.0, 1.0)
    target_sum = len(h_values) / 2.0
    current_sum = float(np.sum(h_values))
    if current_sum <= 0:
        return np.full(len(h_values), 0.5, dtype=float)
    h_values = h_values * (target_sum / current_sum)
    return np.clip(h_values, 0.0, 1.0)

def objective(h_values):
    h_values = project_feasible(h_values)
    return compute_c5(h_values)

def run(seed=42, budget_s=1000, **kwargs):
    np.random.seed(seed)
    h0 = np.asarray(initial_h_values, dtype=float).copy()
    h0 = project_feasible(h0)
    result = minimize(objective, h0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(h0))
    h_best = project_feasible(result.x if result.success else h0)
    n_points = len(h_best)
    c5_bound = compute_c5(h_best)
    evaluate_erdos_solution(h_best, c5_bound, n_points)
    return h_best, c5_bound, n_points
```

If you change the template, keep the same contract and global-variable assumptions.

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

{state_ctx}
{construction_section}
{code_section}
'''


def discover_erdos_min_overlap(
    *,
    backend_type: str = "local_inference",
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    tokenizer_model_name: str | None = None,
    local_model_path: str | None = None,
    renderer_name: str | None = "qwen3_instruct",
    sampler_type: str = "puct",
    num_steps: int = 10,
    group_size: int = 1,
    groups_per_batch: int = 1,
    num_cpus_per_task: int = 1,
    hta_commit_horizon: int = 1,
    map_elites_num_islands: int = 4,
    map_elites_cells_per_dim: int = 4,
    map_elites_migration_interval: int = 5,
    map_elites_migration_top_k: int = 1,
    experiment_name: str | None = None,
    wandb_project: str | None = None,
):
    if experiment_name is None:
        experiment_name = f"erdos-min-overlap-{sampler_type}-{backend_type}"
    config = DiscoverConfig(
        env_type=ErdosMinOverlapEnv,
        problem_type="",
        num_cpus_per_task=num_cpus_per_task,
        eval_timeout=530,
        experiment_name=experiment_name,
        wandb_project=wandb_project,
        backend_type=backend_type,
        model_name=model_name,
        tokenizer_model_name=tokenizer_model_name,
        local_model_path=local_model_path,
        renderer_name=renderer_name,
        sampler_type=sampler_type,
        hta_commit_horizon=hta_commit_horizon,
        num_epochs=num_steps,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        map_elites_num_islands=map_elites_num_islands,
        map_elites_cells_per_dim=map_elites_cells_per_dim,
        map_elites_migration_interval=map_elites_migration_interval,
        map_elites_migration_top_k=map_elites_migration_top_k,
    )
    discover(config)


async def discover_erdos_min_overlap_async(
    *,
    backend_type: str = "local_inference",
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    tokenizer_model_name: str | None = None,
    local_model_path: str | None = None,
    renderer_name: str | None = "qwen3_instruct",
    sampler_type: str = "puct",
    num_steps: int = 10,
    group_size: int = 1,
    groups_per_batch: int = 1,
    num_cpus_per_task: int = 1,
    hta_commit_horizon: int = 1,
    map_elites_num_islands: int = 4,
    map_elites_cells_per_dim: int = 4,
    map_elites_migration_interval: int = 5,
    map_elites_migration_top_k: int = 1,
    experiment_name: str | None = None,
    wandb_project: str | None = None,
):
    if experiment_name is None:
        experiment_name = f"erdos-min-overlap-{sampler_type}-{backend_type}"
    config = DiscoverConfig(
        env_type=ErdosMinOverlapEnv,
        problem_type="",
        num_cpus_per_task=num_cpus_per_task,
        eval_timeout=530,
        experiment_name=experiment_name,
        wandb_project=wandb_project,
        backend_type=backend_type,
        model_name=model_name,
        tokenizer_model_name=tokenizer_model_name,
        local_model_path=local_model_path,
        renderer_name=renderer_name,
        sampler_type=sampler_type,
        hta_commit_horizon=hta_commit_horizon,
        num_epochs=num_steps,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        map_elites_num_islands=map_elites_num_islands,
        map_elites_cells_per_dim=map_elites_cells_per_dim,
        map_elites_migration_interval=map_elites_migration_interval,
        map_elites_migration_top_k=map_elites_migration_top_k,
    )
    await discover_impl(config)
    

if __name__ == "__main__":
    discover_erdos_min_overlap()
