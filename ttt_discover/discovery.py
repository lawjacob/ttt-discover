import asyncio
import logging
import os
from typing import Literal

import chz
from ttt_discover.tinker_utils.dataset_builder import Environment
from ttt_discover.environments.utils.cpu_scheduler import CpuScheduler
import ttt_discover.tinker_utils.misc_utils as misc_utils
from ttt_discover.rl.train import Config, main
from ttt_discover.tinker_utils.dataset_builder import DatasetConfig, get_single_problem_dataset_builder

logger = logging.getLogger(__name__)


@chz.chz
class DiscoverConfig:
    """Simple config for discovery with RL training or local inference-only search."""

    # Model config
    model_name: str = "openai/gpt-oss-120b"
    lora_rank: int = 32
    renderer_name: str | None = "gpt_oss_high_reasoning"
    backend_type: Literal["tinker_train", "local_inference"] = "tinker_train"
    local_model_path: str | None = None
    local_max_new_tokens: int = 2048
    local_device_map: str = "auto"
    save_every: int = 2

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 64
    learning_rate: float = 4e-5
    num_epochs: int = 50
    temperature: float = 1.0
    kl_penalty_coef: float = 0.1
    phase1_max_tokens: int = 26000  # Two-phase sampling: total prompt + thinking token budget

    # Misc config
    experiment_name: str | None = None
    wandb_project: str | None = "tinker-cookbook"

    # Environment-specific
    env_type: str = Environment
    problem_type: str = "26"
    num_cpus_per_task: int = 0
    eval_timeout: int = 1000
    sampler_type: Literal["puct", "hta"] = "hta"
    hta_num_niches: int = 32
    hta_alpha_step: float = 0.05
    hta_stagnation_window: int = 15
    hta_inter_fraction_floor: float = 0.2
    hta_inter_fraction_ceiling: float = 0.8


def init_ray(num_cpus_per_task: int, env_type: str):
    import ray

    if not ray.is_initialized():
        ray.init()
    else:
        if env_type.__name__ != "AhcEnv":
            ray.init("auto")

    try:
        # Try to get existing actor by name
        _scheduler = ray.get_actor("cpu_scheduler")
        print("Found existing cpu_scheduler actor.")
    except ValueError:
        # If not found, create a new one
        print("Creating new cpu_scheduler actor.")
        _scheduler = CpuScheduler.options(
            name="cpu_scheduler",
            lifetime="detached",
        ).remote(
            num_cpus_per_task=num_cpus_per_task,
            num_persistent_workers=0,
        )


async def discover_impl(config: DiscoverConfig):
    """Convert discover config to full config and run training."""

    # Previously restricted to GPT-OSS models; allow arbitrary HF/local models for Colab/testing.
    # Keep a soft warning to remind users that untested models may require compatible renderers.
    if config.model_name not in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}:
        logger.warning("Using non-GPT-OSS model '%s'; ensure the renderer matches the model formatting.", config.model_name)

    # Ray is needed to dispatch jobs across cpus
    if config.num_cpus_per_task > 0:
        init_ray(config.num_cpus_per_task, config.env_type)

    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logging.NullHandler())

    renderer_name = config.renderer_name

    # create log path if it doesn't exist
    log_path = f"./tinker_log/{config.experiment_name}"
    log_file = os.path.join(log_path, "train.log")

    # Resolve env_name -> env type and build dataset
    dataset_config = DatasetConfig(
        env_type=config.env_type,
        problem_type=config.problem_type,
        batch_size=config.groups_per_batch,
        group_size=config.group_size,
        model_name_for_tokenizer=config.local_model_path or config.model_name,
        renderer_name=renderer_name,
        num_cpus_per_task=config.num_cpus_per_task,
        eval_timeout=config.eval_timeout,
        log_path=log_path,
        sampler_type=config.sampler_type,
        hta_num_niches=config.hta_num_niches,
        hta_alpha_step=config.hta_alpha_step,
        hta_stagnation_window=config.hta_stagnation_window,
        hta_inter_fraction_floor=config.hta_inter_fraction_floor,
        hta_inter_fraction_ceiling=config.hta_inter_fraction_ceiling,
    )
    dataset_builder = get_single_problem_dataset_builder(dataset_config)

    rl_config = Config(
        env_type=dataset_config.env_type,
        problem_type=config.problem_type,
        learning_rate=config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model_name,
        lora_rank=config.lora_rank,
        temperature=config.temperature,
        wandb_project=config.wandb_project,
        wandb_name=config.experiment_name,
        log_path=log_path,
        load_checkpoint_path=None,
        kl_penalty_coef=config.kl_penalty_coef,
        num_substeps=1,
        save_every=config.save_every,
        num_epochs=config.num_epochs,
        loss_fn="importance_sampling",
        adv_estimator="entropic_adaptive_beta",
        adv_estimator_beta=2.0, # Unused with entropic_adaptive_beta
        remove_constant_reward_groups=True,
        phase1_max_tokens=config.phase1_max_tokens,
        local_model_path=config.local_model_path,
        backend_type=config.backend_type,
        local_max_new_tokens=config.local_max_new_tokens,
        local_device_map=config.local_device_map,
    )

    misc_utils.check_log_dir(log_path, behavior_if_exists="resume")
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a", force=True)
    logger.info("Logging to %s", log_file)

    # Run training
    await main(rl_config)
    
def discover(config: DiscoverConfig):
    asyncio.run(discover_impl(config))
