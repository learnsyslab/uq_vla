import abc
from typing import Literal

import torch
from torch import Tensor


class CondProbPath(abc.ABC):
    """
    Abstract base class for conditional probability paths used in flow matching.

    Defines the interface for sampling from a time-dependent conditional probability
    path and computing the corresponding conditional velocity field.
    """
    @abc.abstractmethod
    def sample(self, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Sample x_t from the conditional probability p_t(x|x_1).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def velocity(self, x_t: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Compute the conditional velocity vector u_t(x|x_1).
        """
        raise NotImplementedError

    def get_vel_diff_scaling_factor(self, t: Tensor) -> Tensor:
        """
        Compute the time-dependent scaling factor used to weight velocity differences 
        in the intermediate velocity difference score.
        """
        raise NotImplementedError


class OTCondProbPath(CondProbPath):
    """
    Optimal transport-based conditional path using linear interpolation.
    """
    def __init__(self, sigma_min: float = 0):
        super().__init__()
        self.sigma_min = sigma_min

    def sample(self, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Sample x_t from the conditional probability p_t(x|x_1) based on optimal
        transport interpolant.
        """
        # Sample noise prior x_0 ~ N(0, I).
        x_0 = torch.randn_like(
            x_1,
            dtype=x_1.dtype,
            device=x_1.device,
        )
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    def velocity(self, x_t: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Compute velocity u_t(x|x_1) of the optimal transport conditional vector field.
        """
        return (x_1 - (1 - self.sigma_min) * x_t) / (1 - (1 - self.sigma_min) * t)

    def get_vel_diff_scaling_factor(self, t: Tensor) -> Tensor:
        """
        Compute the time-dependent scaling factor used to weight velocity differences 
        in the intermediate velocity difference score.
        """
        return t / (1 - t)


class VPDiffusionCondProbPath(CondProbPath):
    """
    Variance-Preserving Diffusion conditional path with standard linear beta schedule.
    """
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def get_alpha(self, t: Tensor) -> Tensor:
        """Compute the alpha value for the variance-preserving diffusion path."""
        # Convert t so that s = 0 means clean data and s = 1 means full noise as
        # usual in Diffusion
        s = 1.0 - t
        # Compute the cumulative noise amount at s
        T = self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * (s ** 2)
        return torch.exp(-0.5 * T)

    def get_sigma(self, t: Tensor) -> Tensor:
        """Compute the sigma value for the variance-preserving diffusion path."""
        # Convert t so that s = 0 means clean data and s = 1 means full noise as
        # usual in Diffusion
        s = 1.0 - t
        # Compute the cumulative noise amount at s
        T = self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * (s ** 2)
        return torch.sqrt(1.0 - torch.exp(-T))

    def get_beta(self, t: Tensor) -> Tensor:
        """Compute the beta value for the variance-preserving diffusion path."""
        # Convert t so that s = 0 means clean data and s = 1 means full noise as
        # usual in Diffusion
        s = 1.0 - t
        return self.beta_min + s * (self.beta_max - self.beta_min)

    def get_dalpha(self, t: Tensor) -> Tensor:
        """Compute the derivative of alpha with respect to t."""
        # Convert t so that s = 0 means clean data and s = 1 means full noise as
        # usual in Diffusion
        s = 1.0 - t
        T = self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * (s ** 2)
        beta = self.get_beta(t)
        return 0.5 * beta * torch.exp(-0.5 * T)

    def get_dsigma(self, t: Tensor) -> Tensor:
        """Compute the derivative of sigma with respect to t."""
        # Convert t so that s = 0 means clean data and s = 1 means full noise as
        # usual in Diffusion
        s = 1.0 - t
        T = self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * (s ** 2)
        beta = self.get_beta(t)
        return -0.5 * beta * torch.exp(-T) / self.get_sigma(t)

    def sample(self, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Sample x_t from the conditional probability p_t(x|x_1) based on variance-
        preserving diffusion path.
        """
        # Compute how much of the original data remains
        alpha = self.get_alpha(t)
        # Compute how much new noise is added
        sigma = self.get_sigma(t)
        # Mix original data with noise
        epsilon = torch.randn_like(
            x_1,
            dtype=x_1.dtype,
            device=x_1.device,
        )
        return alpha * x_1 + sigma * epsilon

    def velocity(self, x_t: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Compute velocity u_t(x|x_1) of the variance-preserving diffusion conditional
        vector field.
        """
        alpha = self.get_alpha(t)
        sigma = self.get_sigma(t)
        # Compute the derivatives of alpha and sigma with respect to t
        d_alpha = self.get_dalpha(t)
        d_sigma = self.get_dsigma(t)

        return (d_sigma / sigma) * (x_t - alpha * x_1) + d_alpha * x_1

    def get_vel_diff_scaling_factor(self, t: Tensor) -> Tensor:
        """
        Compute the time-dependent scaling factor used to weight velocity differences
        in the intermediate velocity difference score.
        """
        return 2.0 / self.get_beta(t)


def make_cond_prob_path(cond_vf_type: Literal["ot", "vp"], **kwargs) -> CondProbPath:
    """
    Factory function to create a conditional probability path based on the specifie type.

    Args:
        cond_vf_type: Type of conditional vector field ("ot" or "vp").
        **kwargs: Additional parameters for the conditional path.

    Returns:
        An instance of a subclass of CondProbPath.
    """
    if cond_vf_type == "ot":
        return OTCondProbPath(
            sigma_min=kwargs.get("sigma_min", 0.0)
        )
    elif cond_vf_type == "vp":
        return VPDiffusionCondProbPath(
            beta_min=kwargs.get("beta_min", 0.1),
            beta_max=kwargs.get("beta_max", 20.0),
        )
    else:
        raise ValueError(f"Unknown conditional vector field type {cond_vf_type}.")