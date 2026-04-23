from abc import ABC, abstractmethod

import numpy as np
import torch
import gymnasium as gym


class LyapunovWrapper(gym.Wrapper, ABC):
    """
    Base wrapper that replaces the native environment reward with:
        R_t = Φ(s_t) - Φ(s_{t+1})

    Subclasses must implement phi (numpy, used for reward) and
    phi_torch (torch, used by CriticB).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_obs: np.ndarray | None = None

    @staticmethod
    @abstractmethod
    def phi(obs: np.ndarray) -> float:
        """Lyapunov function value for a single observation (numpy)."""
        ...

    @staticmethod
    @abstractmethod
    def phi_torch(obs: torch.Tensor) -> torch.Tensor:
        """Lyapunov function value for a batch of observations (torch).
        Args:  obs (..., obs_dim)
        Returns: (..., 1)
        """
        ...

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, _native_reward, terminated, truncated, info = self.env.step(action)
        phi_prev = self.__class__.phi(self._prev_obs)
        phi_next = self.__class__.phi(obs)
        reward = float(phi_prev - phi_next)
        info["phi_prev"] = phi_prev
        info["phi_next"] = phi_next
        self._prev_obs = obs.copy()
        return obs, reward, terminated, truncated, info
