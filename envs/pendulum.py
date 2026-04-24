import numpy as np
import torch
import gymnasium as gym

from envs.base import LyapunovWrapper


class LyapunovPendulum(LyapunovWrapper):
    """
    Pendulum-v1 with Lyapunov reward R_t = Φ(s_t) - Φ(s_{t+1}).

    Lyapunov function: mechanical energy relative to the upright equilibrium.
        Φ(s) = (1 - cos θ) + 0.5 θ̇²
    Zero at the upright equilibrium (θ=0, θ̇=0), positive everywhere else.
    """

    obs_dim = 3
    act_dim = 1
    action_clip = (-2.0, 2.0)

    def __init__(self, render_mode=None):
        super().__init__(gym.make("Pendulum-v1", render_mode=render_mode))

    @staticmethod
    def phi(obs: np.ndarray) -> float:
        cos_th, _, theta_dot = obs[0], obs[1], obs[2]
        return float((1.0 - cos_th) + 0.5 * theta_dot ** 2)

    @staticmethod
    def phi_terms(obs: np.ndarray) -> dict[str, float]:
        cos_th, _, theta_dot = obs[0], obs[1], obs[2]
        return {
            "position": float(1.0 - cos_th),
            "velocity": float(0.5 * theta_dot ** 2),
        }

    @staticmethod
    def phi_torch(obs: torch.Tensor) -> torch.Tensor:
        cos_th = obs[..., 0:1]
        theta_dot = obs[..., 2:3]
        return (1.0 - cos_th) + 0.5 * theta_dot ** 2

    @staticmethod
    def state_grid(n: int = 50):
        theta = np.linspace(-np.pi, np.pi, n)
        theta_dot = np.linspace(-8.0, 8.0, n)
        TH, TD = np.meshgrid(theta, theta_dot)
        states = np.stack([np.cos(TH), np.sin(TH), TD], axis=-1).reshape(-1, 3).astype(np.float32)
        return TH, TD, states
    
    def shaping_reward(self, obs, action, next_obs, terminated) -> dict[str, float]:
        return {
            "action_penalty": -0.001 * float(np.sum(action ** 2)),
        }
