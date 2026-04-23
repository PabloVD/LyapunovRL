import numpy as np
import gymnasium as gym
from lyapunov.functions import phi_np


class LyapunovPendulum(gym.Wrapper):
    """
    Wraps Pendulum-v1, replacing the native reward with the Lyapunov-based reward:
        R_t = Φ(s_t) - Φ(s_{t+1})
    Positive when the system moves toward the upright equilibrium.
    """

    def __init__(self, render_mode=None):
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        super().__init__(env)
        self._prev_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, _native_reward, terminated, truncated, info = self.env.step(action)
        reward = float(phi_np(self._prev_obs) - phi_np(obs))
        info["phi_prev"] = float(phi_np(self._prev_obs))
        info["phi_next"] = float(phi_np(obs))
        self._prev_obs = obs.copy()
        return obs, reward, terminated, truncated, info
