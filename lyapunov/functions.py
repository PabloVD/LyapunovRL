import numpy as np
import torch


def phi_np(states: np.ndarray) -> np.ndarray:
    """
    Lyapunov function: mechanical energy relative to upright equilibrium.
    Φ(s) = (1 - cos θ) + 0.5 θ̇²

    Args:
        states: (..., 3) array with components [cos θ, sin θ, θ̇]
    Returns:
        (...,) array of Φ values
    """
    cos_th = states[..., 0]
    theta_dot = states[..., 2]
    return (1.0 - cos_th) + 0.5 * theta_dot ** 2


def phi_torch(states: torch.Tensor) -> torch.Tensor:
    """
    Lyapunov function for torch tensors. Same formula as phi_np.

    Args:
        states: (..., 3) tensor with components [cos θ, sin θ, θ̇]
    Returns:
        (..., 1) tensor of Φ values (column vector for critic compatibility)
    """
    cos_th = states[..., 0:1]
    theta_dot = states[..., 2:3]
    return (1.0 - cos_th) + 0.5 * theta_dot ** 2
