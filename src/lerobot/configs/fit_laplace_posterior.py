from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import LaplaceConfig


@dataclass
class FitLaplacePosteriorPipelineConfig:
    laplace: LaplaceConfig = field(default_factory=LaplaceConfig)
    policy: PreTrainedConfig | None = None
    dataset: DatasetConfig | None = None

    seed: int | None = None
    output_dir: Path | None = None

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError(
                "No pretrained policy path was provided for fitting the Laplace posterior."
            )

        if not self.output_dir:
            self.output_dir = Path(policy_path)

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
