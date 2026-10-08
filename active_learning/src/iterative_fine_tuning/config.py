"""Configuration objects for iterative active-learning fine-tuning."""

from __future__ import annotations

import builtins
import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import LaplaceConfig

from .constants import ITERATIVE_CONFIG_NAME, STAGE_SELECT, STAGE_TRAIN, VALID_STAGES
from .env import (
    ensure_no_unresolved_env_vars,
    expand_value,
    maybe_load_env_file,
    resolve_local_path,
    resolve_model_reference,
)


@dataclass
class PathConfig:
    env_file: str | None = ".env"
    run_root: str = "outputs/active_learning"
    checkpoint_root: str | None = None
    run_name: str = "default_run"


@dataclass
class IterationConfig:
    total_rounds: int = 1
    episodes_per_round: int = 10
    bootstrap_selection_paths: list[str] = field(default_factory=list)
    warm_start_from_previous_round: bool = True


@dataclass
class ReplayMixingConfig:
    history_tasks: dict[str, list[int]] | None
    # Optional explicit history episode ids; overrides the history_tasks filter when set
    # (needed to replay specific episodes for single-task benchmarks like Push-T).
    history_episode_ids: list[int] | None = None
    history_fraction: float = 0.5
    new_fraction: float = 0.5
    sampling_mode: str = "weighted_pools"
    samples_per_epoch: int | None = None
    replacement: bool = True
    # Decode the selected ("new") episodes once into shared memory instead of re-decoding their frames
    # on every draw. They are half of every batch, so this removes half the decode load; see
    # iterative_fine_tuning/frame_cache.py.
    cache_new_pool_frames: bool = False

    def validate(self) -> None:
        if self.history_tasks is None:
            raise ValueError("replay.history_tasks must be configured explicitly.")
        if self.sampling_mode not in {"weighted_pools", "uniform_all_frames"}:
            raise ValueError(
                "sampling_mode must be one of {'weighted_pools', 'uniform_all_frames'}."
            )
        vals = {
            "history_fraction": self.history_fraction,
            "new_fraction": self.new_fraction,
        }
        for name, value in vals.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if sum(vals.values()) <= 0:
            raise ValueError("At least one replay fraction must be positive.")
        if self.samples_per_epoch is not None and self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive when provided.")


@dataclass
class AMFConfig:
    """Settings for the AMF posterior-covariance selection strategy."""

    noise: float = 1.0e-2  # Ridge term sigma in (A A^T + sigma I)^-1
    frames_per_demo: int = 24  # Frames embedded per demonstration
    num_time_samples: int = 1  # Flow-matching (time, noise) draws averaged per frame
    prefill_demos_per_task: int = 1  # Demonstrations acquired per task before the criterion runs
    normalize_embeddings: bool = True  # Rescale the kernel so `noise` is policy-independent
    max_demos_per_task: int | None = None  # Cap on proxy demonstrations, keeping the most recent
    scope: str = "action_out_proj"  # Last layer differentiated for the embeddings

    def validate(self) -> None:
        if self.noise <= 0:
            raise ValueError("selection.amf.noise must be positive.")
        if self.frames_per_demo <= 0:
            raise ValueError("selection.amf.frames_per_demo must be positive.")
        if self.num_time_samples <= 0:
            raise ValueError("selection.amf.num_time_samples must be positive.")
        if self.prefill_demos_per_task < 0:
            raise ValueError("selection.amf.prefill_demos_per_task must be non-negative.")
        if self.max_demos_per_task is not None and self.max_demos_per_task <= 0:
            raise ValueError("selection.amf.max_demos_per_task must be positive when provided.")


@dataclass
class SelectionConfig:
    ensemble_model_paths: list[str]
    strategy: str = "top_k"
    uncertainty_method: str = "cross_bayesian"
    scorer_type: str = "ensemble"
    scoring_metric: str = "inter_vel_diff"
    num_action_samples: int = 5
    observation_batch_size: int = 8
    laplace: LaplaceConfig = field(default_factory=LaplaceConfig)
    amf: AMFConfig = field(default_factory=AMFConfig)
    candidate_tasks: dict[str, list[int]] | None = None
    max_candidates_per_task: int | None = None # Only used for "top_k" and "balanced_top_k" strategies
    max_selected_per_task: int | None = None # Only used for "constrained_top_k" strategy
    task_selection_temperature: float = 1.0
    task_weighted_episode_selection: str = "top_k"
    device: str = "cuda:0"
    seed: int = 0
    # Path to a precomputed episode ranking JSON; required when strategy == "predefined_ranking".
    predefined_ranking_path: str | None = None

    def validate(self) -> None:
        valid_strategies = {
            "top_k",
            "balanced_top_k",
            "constrained_top_k",
            "task_weighted_top_k",
            "random",
            "predefined_ranking",
            "amf",
        }
        if self.strategy not in valid_strategies:
            raise ValueError(f"Unknown selection strategy {self.strategy!r}.")
        if self.strategy not in {"predefined_ranking", "amf"} and not self.ensemble_model_paths:
            raise ValueError("selection.ensemble_model_paths must be configured explicitly.")
        if self.strategy == "predefined_ranking" and not self.predefined_ranking_path:
            raise ValueError(
                "selection.predefined_ranking_path must be set when strategy='predefined_ranking'."
            )
        if self.strategy == "amf":
            # AMF derives its own criterion from loss-gradient embeddings and ignores the
            # uncertainty-sampler settings below.
            self.amf.validate()
            return
        if self.uncertainty_method not in {"cross_bayesian", "entropy", "ace"}:
            raise ValueError(f"Unknown uncertainty method {self.uncertainty_method!r}.")
        if self.scorer_type not in {"ensemble", "laplace"}:
            raise ValueError(f"Unknown scorer type {self.scorer_type!r}.")
        if self.uncertainty_method in {"entropy", "ace"} and self.scorer_type != "ensemble":
            raise ValueError(
                f"{self.uncertainty_method} selection does not use scorer_type; "
                "please keep scorer_type='ensemble'."
            )
        if self.scoring_metric in ("inter_vel_diff", "inter_vel_diff_2way") and self.scorer_type != "ensemble":
            raise ValueError(f"{self.scoring_metric} currently requires scorer_type='ensemble'.")
        if self.strategy == "constrained_top_k" and self.max_selected_per_task is None:
            raise ValueError("constrained_top_k requires max_selected_per_task to be set.")
        if self.observation_batch_size <= 0:
            raise ValueError("observation_batch_size must be positive.")
        if self.task_selection_temperature < 0:
            raise ValueError("task_selection_temperature must be non-negative.")
        if self.task_weighted_episode_selection not in {"top_k", "uniform_random"}:
            raise ValueError(
                "task_weighted_episode_selection must be one of {'top_k', 'uniform_random'}."
            )


@dataclass
class ExecutionConfig:
    stages: tuple[str, ...] = (STAGE_SELECT, STAGE_TRAIN)
    resume: bool = False
    overwrite_existing_round: bool = False
    save_candidate_scores: bool = True

    def validate(self) -> None:
        invalid = [stage for stage in self.stages if stage not in VALID_STAGES]
        if invalid:
            raise ValueError(f"Invalid execution stages: {invalid}.")


@dataclass
class EnsembleTrainingConfig:
    """Where each ensemble member trains.

    `member_devices` holds one entry per member, e.g. `["cuda:0", "cuda:1"]`. An entry naming several
    GPUs, `"cuda:0,1,2,3"`, trains that member data-parallel over them (worker launched through
    `accelerate launch`, DDP); `batch_size` and `num_workers` are then per process.
    """

    parallel_members: bool = False
    member_devices: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.parallel_members and not self.member_devices:
            raise ValueError(
                "ensemble_training.member_devices must be configured when parallel_members is enabled."
            )
        if any(not str(device).strip() for device in self.member_devices):
            raise ValueError("ensemble_training.member_devices must not contain empty device strings.")


@dataclass
class SeedPairSpec:
    """Everything that differs between the seed pairs of one setup (see `seed_pairs`).

    `members` are the pretraining seeds of the two ensemble members; they fill `${SEED_A}` and
    `${SEED_B}` in the config's paths, and the pair's name fills `${SEED_PAIR}`. The other fields
    replace `seed`, `selection.seed` and `replay.history_episode_ids` when set.
    """

    members: list[int]
    seed: int | None = None
    selection_seed: int | None = None
    history_episode_ids: list[int] | None = None


SEED_PAIR_VARS = ("SEED_PAIR", "SEED_A", "SEED_B")


@dataclass
class IterativeFineTuningConfig:
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    paths: PathConfig = field(default_factory=PathConfig)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    replay: ReplayMixingConfig = field(
        default_factory=lambda: ReplayMixingConfig(history_tasks=None)
    )
    selection: SelectionConfig = field(
        default_factory=lambda: SelectionConfig(ensemble_model_paths=[])
    )
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    ensemble_training: EnsembleTrainingConfig = field(default_factory=EnsembleTrainingConfig)
    # One config per setup, one job per seed pair: `seed_pairs` lists every pair the setup was run
    # with, and `seed_pair` (or the SEED_PAIR environment variable) picks the one this job runs.
    # validate() then rewrites the config into exactly the single-pair config it stands for and
    # empties `seed_pairs`, so the run's config snapshot names one pair only.
    seed_pairs: dict[str, SeedPairSpec] = field(default_factory=dict)
    seed_pair: str | None = None
    job_name: str | None = None
    reinitialize_selected_layers: bool = False
    seed: int | None = 1000
    num_workers: int = 4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    steps: int = 10_000
    eval_freq: int = 0
    log_freq: int = 200
    save_checkpoint: bool = True
    save_freq: int = 2_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    fixed_learning_rate: float | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    rename_map: dict[str, str] = field(default_factory=dict)
    initial_model_reference: str | None = field(init=False, default=None)

    def validate(self) -> None:
        base_dir = Path.cwd()
        maybe_load_env_file(self.paths.env_file, base_dir=base_dir)
        self._apply_seed_pair()
        self.paths.run_name = expand_value(self.paths.run_name)
        ensure_no_unresolved_env_vars(self.paths.run_name)
        if self.job_name:
            self.job_name = expand_value(self.job_name)
            ensure_no_unresolved_env_vars(self.job_name)

        # `dataset.root` is the dataset directory itself. Left unset, LeRobot resolves it to
        # $HF_LEROBOT_HOME/<repo_id> (default ~/.cache/huggingface/lerobot/<repo_id>).
        dataset_root = getattr(self.dataset, "root", None)
        if dataset_root:
            self.dataset.root = str(resolve_local_path(dataset_root, base_dir=base_dir))

        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.initial_model_reference = resolve_model_reference(policy_path, base_dir=base_dir)
        elif self.policy is not None and self.policy.pretrained_path is not None:
            self.initial_model_reference = str(
                resolve_local_path(self.policy.pretrained_path, base_dir=base_dir)
            )

        if self.policy is None:
            raise ValueError(
                "Policy is not configured. Please specify a pretrained policy with `--policy.path`."
            )
        if self.initial_model_reference is None:
            raise ValueError("Iterative fine-tuning requires a starting model reference.")
        if self.dataset.libero_tasks is not None and self.dataset.pusht_tasks is not None:
            raise ValueError(
                "Configure only one benchmark task map: dataset.libero_tasks or dataset.pusht_tasks."
            )

        self.replay.validate()
        self.selection.validate()
        self.execution.validate()
        self.ensemble_training.validate()

        if self.iteration.total_rounds <= 0:
            raise ValueError("iteration.total_rounds must be positive.")
        if self.iteration.episodes_per_round <= 0:
            raise ValueError("iteration.episodes_per_round must be positive.")
        if self.fixed_learning_rate is not None and self.fixed_learning_rate <= 0:
            raise ValueError("fixed_learning_rate must be positive when provided.")
        if STAGE_TRAIN in self.execution.stages and not self.save_checkpoint:
            raise ValueError(
                "save_checkpoint must be enabled when the train stage is active so iterative rounds can warm-start."
            )
        if not self.job_name:
            self.job_name = self.paths.run_name

        if self.use_policy_training_preset:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()
        elif self.optimizer is None:
            raise ValueError(
                "optimizer must be configured when use_policy_training_preset is False."
            )

        if self.selection.candidate_tasks is None:
            self.selection.candidate_tasks = copy.deepcopy(
                self.dataset.libero_tasks or self.dataset.pusht_tasks
            )
        if self.selection.candidate_tasks is None:
            raise ValueError(
                "selection.candidate_tasks is not configured and neither "
                "dataset.libero_tasks nor dataset.pusht_tasks is set."
            )

        if (
            self.selection.strategy not in {"predefined_ranking", "amf"}
            and self.selection.uncertainty_method == "cross_bayesian"
            and self.selection.scorer_type == "ensemble"
            and self.selection.scoring_metric in ("inter_vel_diff", "inter_vel_diff_2way")
            and len(self.resolved_ensemble_model_paths()) < 2
        ):
            raise ValueError(
                f"selection.scoring_metric='{self.selection.scoring_metric}' requires at least two ensemble model paths."
            )

    def _apply_seed_pair(self) -> None:
        if not self.seed_pairs:
            return
        name = self.seed_pair or os.environ.get("SEED_PAIR")
        if not name:
            raise ValueError(
                f"This config lists seed pairs {sorted(self.seed_pairs)}; choose one with "
                "--seed_pair=<name> or SEED_PAIR=<name>."
            )
        if name not in self.seed_pairs:
            raise ValueError(f"Unknown seed pair {name!r}; this config lists {sorted(self.seed_pairs)}.")
        spec = self.seed_pairs[name]
        if len(spec.members) != 2:
            raise ValueError(f"seed_pairs.{name}.members must name exactly two seeds, got {spec.members}.")
        values = dict(zip(SEED_PAIR_VARS, (name, *(str(m) for m in spec.members))))
        # Substituted directly rather than through the environment, so a stale SEED_A exported by
        # the calling shell cannot leak in. SEED_A/SEED_B are exported as well for any later
        # ${SEED_A} expansion; SEED_PAIR is not, or the next config this process loads would
        # silently pick up this pair instead of failing for want of one.
        os.environ.update({k: v for k, v in values.items() if k != "SEED_PAIR"})

        def sub(value: str) -> str:
            for var, val in values.items():
                value = value.replace("${" + var + "}", val)
            return value

        self.paths.run_name = sub(self.paths.run_name)
        if self.paths.checkpoint_root:
            self.paths.checkpoint_root = sub(self.paths.checkpoint_root)
        if self.job_name:
            self.job_name = sub(self.job_name)
        if self.policy is not None and self.policy.pretrained_path is not None:
            self.policy.pretrained_path = Path(sub(str(self.policy.pretrained_path)))
        self.selection.ensemble_model_paths = [sub(p) for p in self.selection.ensemble_model_paths]
        if spec.seed is not None:
            self.seed = spec.seed
        if spec.selection_seed is not None:
            self.selection.seed = spec.selection_seed
        if spec.history_episode_ids is not None:
            self.replay.history_episode_ids = list(spec.history_episode_ids)
        self.seed_pair = name
        self.seed_pairs = {}

    def run_root(self) -> Path:
        return resolve_local_path(self.paths.run_root, base_dir=Path.cwd())

    def run_dir(self) -> Path:
        return self.run_root() / self.paths.run_name

    def checkpoint_root(self) -> Path:
        if self.paths.checkpoint_root is None:
            return self.run_root()
        return resolve_local_path(self.paths.checkpoint_root, base_dir=Path.cwd())

    def checkpoint_run_dir(self) -> Path:
        return self.checkpoint_root() / self.paths.run_name

    def bootstrap_selection_paths(self) -> list[Path]:
        return [
            resolve_local_path(path, base_dir=Path.cwd(), must_exist=True)
            for path in self.iteration.bootstrap_selection_paths
        ]

    def resolved_ensemble_model_paths(self) -> list[str]:
        return list(
            dict.fromkeys(
                resolve_model_reference(path, base_dir=Path.cwd())
                for path in self.selection.ensemble_model_paths
            )
        )

    def resolved_member_devices(self, member_count: int) -> list[str]:
        if member_count <= 0:
            return []
        if not self.ensemble_training.parallel_members:
            return []
        if len(self.ensemble_training.member_devices) < member_count:
            raise ValueError(
                "ensemble_training.member_devices must provide at least one device per ensemble member. "
                f"Received {len(self.ensemble_training.member_devices)} devices for {member_count} members."
            )
        return [str(device) for device in self.ensemble_training.member_devices[:member_count]]

    def to_train_config(
        self,
        *,
        output_dir: Path,
        round_index: int,
        policy_cfg: PreTrainedConfig,
    ) -> TrainPipelineConfig:
        optimizer = copy.deepcopy(self.optimizer)
        scheduler = copy.deepcopy(self.scheduler)
        if self.fixed_learning_rate is not None:
            if optimizer is None:
                raise ValueError(
                    "fixed_learning_rate requires an optimizer configuration to be available."
                )
            optimizer.lr = self.fixed_learning_rate
            if scheduler is not None:
                # Setting a fixed LR necessarily discards the schedule -- warmup and decay
                # would both override it. This is easy to do by accident: the field name
                # suggests it only pins the LR value, and a flat LR trains without any
                # visible error. Leave `fixed_learning_rate` unset to keep the policy
                # preset's cosine+warmup schedule (whose peak LR is usually the value you
                # were about to pin anyway).
                logging.getLogger(__name__).warning(
                    "fixed_learning_rate=%s discards the %s scheduler: training will run at a "
                    "CONSTANT learning rate with no warmup and no decay.",
                    self.fixed_learning_rate,
                    type(scheduler).__name__,
                )
            scheduler = None

        return TrainPipelineConfig(
            dataset=copy.deepcopy(self.dataset),
            env=copy.deepcopy(self.env),
            policy=copy.deepcopy(policy_cfg),
            output_dir=output_dir,
            job_name=f"{self.job_name}_round_{round_index:03d}",
            resume=False,
            reinitialize_selected_layers=self.reinitialize_selected_layers,
            seed=self.seed,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            steps=self.steps,
            eval_freq=self.eval_freq,
            log_freq=self.log_freq,
            save_checkpoint=self.save_checkpoint,
            save_freq=self.save_freq,
            use_policy_training_preset=self.use_policy_training_preset,
            optimizer=optimizer,
            scheduler=scheduler,
            eval=copy.deepcopy(self.eval),
            wandb=copy.deepcopy(self.wandb),
            rename_map=copy.deepcopy(self.rename_map),
        )

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]

    @classmethod
    def from_pretrained(
        cls: builtins.type["IterativeFineTuningConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "IterativeFineTuningConfig":
        config_ref = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(config_ref).is_dir():
            if ITERATIVE_CONFIG_NAME in os.listdir(config_ref):
                config_file = os.path.join(config_ref, ITERATIVE_CONFIG_NAME)
            else:
                raise FileNotFoundError(f"{ITERATIVE_CONFIG_NAME} not found in {config_ref}")
        elif Path(config_ref).is_file():
            config_file = config_ref
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=config_ref,
                    filename=ITERATIVE_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as exc:
                raise FileNotFoundError(
                    f"{ITERATIVE_CONFIG_NAME} not found on the Hugging Face Hub in {config_ref}"
                ) from exc
        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)
