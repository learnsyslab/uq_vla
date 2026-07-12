import logging

from lerobot.configs import parser
from lerobot.configs.fit_laplace_posterior import FitLaplacePosteriorPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.uncertainty.uncertainty_scoring.laplace_utils.posterior_builder import get_laplace_posterior
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


@parser.wrap()
def main(cfg: FitLaplacePosteriorPipelineConfig):
    # Set global seed
    set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    allowed_policies = {"flow_matching", "smolvla"}
    if cfg.policy.type not in allowed_policies:
        raise ValueError(
            f"eval_uncertainty_estimation.py only supports policy types {allowed_policies}, "
            f"but got '{cfg.policy.type}'."
        )

    logging.info("Loading policy")
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    policy: PreTrainedPolicy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    ).to(device)
    policy.eval()

    # Build preprocessing/postprocessing pipelines for observations/actions
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Fit or load the Laplace posterior
    get_laplace_posterior(
        policy=policy,
        preprocessor=preprocessor,
        laplace_config=cfg.laplace,
        dataset_cfg=cfg.dataset,
    )


if __name__ == "__main__":
    init_logging()
    main()
