import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.common import envs
from lerobot.common.policies.flow_matching.uncertainty.configuration_uncertainty_sampler import (
    UncertaintySamplerConfig,
)
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, VisConfig
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class VisualizeLaplacePipelineConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    vis: VisConfig = field(default_factory=VisConfig)
    dataset: DatasetConfig | None = None
    uncertainty_sampler: UncertaintySamplerConfig = field(default_factory=UncertaintySamplerConfig)

    # Number of Laplace models
    n_laplace_models: int = 5

    seed: int | None = None
    job_name: str | None = None
    output_dir: Path | None = None
    # `show` enables live visualization of the first environment during evaluation
    show: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, visualized policy will be built from scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            vis_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/visualizations") / vis_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def validate(self):
        # vis_types check
        allowed_vis = {"flows", "vector_field", "action_seq"}
        for v in self.vis_types:
            if v not in allowed_vis:
                raise ValueError(
                    f"Unknown visualization type '{v}'. "
                    f"Allowed: {sorted(allowed_vis)}"
                )