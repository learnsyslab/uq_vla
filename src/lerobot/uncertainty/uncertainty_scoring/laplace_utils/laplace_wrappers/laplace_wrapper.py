from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor import PolicyProcessorPipeline


@dataclass
class LaplaceBatch:
    """
    Container for one batch of inputs to a flow matching model including RGB encoder.
    """
    # Interpolated trajectory based on some random noise sample and a target action
    # using an optimal transport conditional probability path.
    interp_traj: Tensor # Shape: (batch_size, horizon, action_dim)
    # Time step along the flow path.
    time: Tensor # Shape: (batch_size,)
    # Input observations of the environment.
    observation: Dict[str, Tensor]
    in_episode_mask: Tensor

    def to(self, *args, **kwargs) -> "LaplaceBatch":
        """
        Return a copy of this FlowMatchingInput with all contained tensors moved or cast.
        """
        return LaplaceBatch(
            interp_traj = self.interp_traj.to(*args, **kwargs),
            time = self.time.to(*args, **kwargs),
            observation = {k: v.to(*args, **kwargs) for k, v in self.observation.items()},
            in_episode_mask = self.in_episode_mask.to(*args, **kwargs)
        )

    def detach(self) -> "LaplaceBatch":
        return LaplaceBatch(
            interp_traj = self.interp_traj.detach(),
            time = self.time.detach(),
            observation = {k: v.detach() for k, v in self.observation.items()},
            in_episode_mask = self.in_episode_mask.detach(),
        )

    def cpu(self) -> "LaplaceBatch":
        return self.to("cpu")

class LaplaceWrapper(nn.Module, ABC):
    def __init__(self, model: nn.Module, config: PreTrainedConfig, scopes: List[str]):
        super().__init__()
        self.model = model
        self.config = config
        self.scopes = scopes
        self.scope_abbr: Dict[str, str] = {}

        self.apply_laplace_scope()

    @property
    def device(self) -> int:
        return self.config.device

    @abstractmethod
    def build_collate_fn(
        self, preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    ) -> Callable[[Dict[str, Tensor]], Tuple[LaplaceBatch, Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch: LaplaceBatch) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def apply_laplace_scope(self, approx_targets: List[str]):
        raise NotImplementedError
