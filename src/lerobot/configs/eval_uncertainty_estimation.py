import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalUncertEstConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import EnvConfig
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    UncertaintySamplerConfig,
)


@dataclass
class EvalUncertaintyEstimationPipelineConfig:
    env: EnvConfig
    eval_uncert_est: EvalUncertEstConfig = field(default_factory=EvalUncertEstConfig)
    policy: PreTrainedConfig | None = None
    uncertainty_sampler: UncertaintySamplerConfig = field(default_factory=UncertaintySamplerConfig)
    dataset: DatasetConfig | None = None

    seed: int | None = None
    job_name: str | None = None
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
                "No pretrained policy path was provided for the uncertainty evaluation."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval_uncertainty_estimation") / eval_dir

        self.validate()

    def validate(self):
        if (
            "cross_bayesian" in self.eval_uncert_est.uncert_est_methods and
            self.uncertainty_sampler.cross_bayesian_sampler.scorer_type == "ensemble" and
            not self.uncertainty_sampler.cross_bayesian_sampler.ensemble_model_paths
        ):
             raise ValueError(
                "Cross-Bayesian uncertainty sampler with scorer_type=ensemble requested but no" \
                "ensemble checkpoint provided."
            )

        if (
            "composed_cross_bayesian" in self.eval_uncert_est.uncert_est_methods and
            self.uncertainty_sampler.composed_cross_bayesian_sampler.scorer_type == "ensemble" and
            not self.uncertainty_sampler.composed_cross_bayesian_sampler.ensemble_model_paths
        ):
            raise ValueError(
                "Composed Cross-Bayesian uncertainty sampler with scorer_type=ensemble requested but no " \
                "ensemble checkpoint provided."
            )

        if (
            "cross_bayesian" in self.eval_uncert_est.uncert_est_methods and
            self.uncertainty_sampler.cross_bayesian_sampler.scorer_type == "laplace" and
            self.dataset is None
        ):
            raise ValueError(
                "Cross-Bayesian uncertainty sampler with scorer_type=laplace requested but no dataset "
                "config for Laplace approximation provided."
            )

        if (
            "composed_cross_bayesian" in self.eval_uncert_est.uncert_est_methods and
            self.uncertainty_sampler.composed_cross_bayesian_sampler.scorer_type == "laplace" and
            self.dataset is None
        ):
            raise ValueError(
                "Composed Cross-Bayesian uncertainty sampler with scorer_type=laplace requested but no " \
                "dataset config for Laplace approximation provided."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
