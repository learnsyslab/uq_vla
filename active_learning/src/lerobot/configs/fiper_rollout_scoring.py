import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.fiper_data_generator.configuration_fiper_rollout_scorer import (
    FiperRolloutScorerConfig,
)


@dataclass
class FiperRolloutScoringPipelineConfig:
    env: envs.EnvConfig
    input_dir: Path | None = None
    fiper_rollout_scorer: FiperRolloutScorerConfig = field(default_factory=FiperRolloutScorerConfig)
    policy: PreTrainedConfig | None = None
    dataset: DatasetConfig | None = None

    seed: int | None = None
    job_name: str | None = None
    output_dir: Path | None = None
    save_videos: bool = True

    start_episode: int | None = None
    end_episode: int | None = None
    allow_existing_output_dir: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, policy for FIPER rollout data scoring will be built from "
                "scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            run_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/fiper_rollout_scoring") / run_dir

        self.validate()

    def validate(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"input_dir does not exist: {self.input_dir}")

        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"input_dir is not a directory: {self.input_dir}")

        if self.output_dir.exists() and not self.allow_existing_output_dir:
            raise FileExistsError(
                f"output_dir already exists: {self.output_dir}. Please remove it or provide a different --output_dir."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
