import numpy as np
import torch
import gymnasium as gym

from envs.base import LyapunovWrapper


class LyapunovPendulum(LyapunovWrapper):
    """
    Pendulum-v1 with Lyapunov reward R_t = Φ(s_t) - Φ(s_{t+1}).

    Lyapunov function: mechanical energy relative to the upright equilibrium.
        Φ(s) = (1 - cos θ) + 0.5 θ̇²
    """

    def __init__(self, render_mode=None):
        super().__init__(gym.make("Pendulum-v1", render_mode=render_mode))

    @staticmethod
    def phi(obs: np.ndarray) -> float:
        cos_th, _, theta_dot = obs[0], obs[1], obs[2]
        return float((1.0 - cos_th) + 0.5 * theta_dot ** 2)

    @staticmethod
    def phi_torch(obs: torch.Tensor) -> torch.Tensor:
        cos_th = obs[..., 0:1]
        theta_dot = obs[..., 2:3]
        return (1.0 - cos_th) + 0.5 * theta_dot ** 2
