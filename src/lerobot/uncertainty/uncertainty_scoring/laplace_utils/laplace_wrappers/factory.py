from typing import List

from lerobot.policies.pretrained import PreTrainedPolicy

from .laplace_wrapper import LaplaceWrapper


def make_laplace_wrapper(policy: PreTrainedPolicy, scopes: List[str]) -> LaplaceWrapper:
    if policy.config.type == "flow_matching":
        from lerobot.uncertainty.uncertainty_scoring.laplace_utils.laplace_wrappers.flow_matching_laplace_wrapper import (
            FlowMatchingLaplaceWrapper,
        )

        return FlowMatchingLaplaceWrapper(
            model=policy.flow_matching,
            config=policy.config,
            scopes=scopes,
        )
    elif policy.config.type == "smolvla":
        from lerobot.uncertainty.uncertainty_scoring.laplace_utils.laplace_wrappers.smolvla_laplace_wrapper import (
            SmolVLALaplaceWrapper,
        )

        return SmolVLALaplaceWrapper(
            model=policy.model,
            config=policy.config,
            scopes=scopes,
        )
    else:
        raise ValueError(f"No laplace wrapper available for policy type '{policy.config.type}'.")
