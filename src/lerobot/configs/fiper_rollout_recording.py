import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.fiper_data_generator.configuration_fiper_rollout_recorder import (
    FiperRolloutRecorderConfig,
)


@dataclass
class FiperRolloutRecordingPipelineConfig:
    env: envs.EnvConfig
    fiper_rollout_recorder: FiperRolloutRecorderConfig = field(default_factory=FiperRolloutRecorderConfig)
    policy: PreTrainedConfig | None = None

    n_calib_episodes: int = 60
    n_test_episodes: int = 240
    # If set, calibration retries stop after this many attempts and the last
    # rollout is saved with its actual success flag. Useful for smoke tests
    # when a policy cannot solve every task reliably.
    max_calibration_attempts_per_task: int | None = None
    # Which environment domains to evaluate. Allowed values: "id" (in-distribution), "ood" (out-of-distribution).
    domains: list[str] = field(default_factory=lambda: ["id", "ood"])
    seed: int | None = None
    job_name: str | None = None
    output_dir: Path | None = None
    save_videos: bool = True

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, policy for FIPER rollout data recording will be built from "
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
            self.output_dir = Path("outputs/fiper_rollout_recording") / run_dir

        self.validate()

    def validate(self):
        pass

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
