import numpy as np
import torch
import gymnasium as gym

from envs.base import LyapunovWrapper

# MountainCarContinuous-v0 physical constants (from Gymnasium source)
_GRAVITY = 0.0025
_GOAL_X = 0.45
# Reference energy: potential energy at the goal position with zero velocity
_E_REF = float(_GRAVITY * np.sin(3.0 * _GOAL_X))


class LyapunovMountainCar(LyapunovWrapper):
    """
    MountainCarContinuous-v0 with Lyapunov reward R_t = Φ(s_t) - Φ(s_{t+1}).

    Lyapunov function: energy deficit relative to the goal height.
        E(x, v) = 0.5 v² + g·sin(3x)          (kinetic + potential energy)
        Φ(s)    = max(0, E_ref − E(x, v))       (zero once car has enough energy)
    where E_ref = g·sin(3·x_goal) is the potential energy at the goal with v=0.

    The reward R_t > 0 whenever the car gains energy toward E_ref, incentivising
    the necessary energy-pumping swing-up before cresting the hill.
    """

    obs_dim = 2
    act_dim = 1
    action_clip = (-1.0, 1.0)

    def __init__(self, render_mode=None):
        super().__init__(gym.make("MountainCarContinuous-v0", render_mode=render_mode))

    @staticmethod
    def phi(obs: np.ndarray) -> float:
        x, v = obs[0], obs[1]
        energy = 0.5 * v ** 2 + _GRAVITY * np.sin(3.0 * x)
        return float(max(0.0, _E_REF - energy))

    @staticmethod
    def phi_terms(obs: np.ndarray) -> dict[str, float]:
        # When both states are inside the deficit region the deltas sum to the
        # Lyapunov reward exactly. At the clamp boundary they diverge slightly.
        x, v = obs[0], obs[1]
        return {
            "kinetic": float(0.5 * v ** 2),
            "potential": float(_GRAVITY * np.sin(3.0 * x)),
        }

    @staticmethod
    def phi_torch(obs: torch.Tensor) -> torch.Tensor:
        x = obs[..., 0:1]
        v = obs[..., 1:2]
        energy = 0.5 * v ** 2 + _GRAVITY * torch.sin(3.0 * x)
        e_ref = torch.tensor(_E_REF, dtype=obs.dtype, device=obs.device)
        return torch.clamp(e_ref - energy, min=0.0)

    @staticmethod
    def state_grid(n: int = 50):
        xs = np.linspace(-1.2, 0.6, n)
        vs = np.linspace(-0.07, 0.07, n)
        XS, VS = np.meshgrid(xs, vs)
        states = np.stack([XS, VS], axis=-1).reshape(-1, 2).astype(np.float32)
        return XS, VS, states
    
    def shaping_reward(self, obs, action, next_obs, terminated) -> dict[str, float]:
        return {
            "action_penalty": -0.1 * float(np.sum(action ** 2)),
            "success_bonus": 100.0 if terminated else 0.0,
        }

