import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from envs.base import LyapunovWrapper

# CartPole physical constants (matching CartPole-v1)
_GRAVITY = 9.8
_MASSCART = 1.0
_MASSPOLE = 0.1
_TOTAL_MASS = _MASSCART + _MASSPOLE
_HALF_LENGTH = 0.5
_POLEMASS_LENGTH = _MASSPOLE * _HALF_LENGTH
_FORCE_MAG = 10.0
_DT = 0.02
_X_THRESHOLD = 2.4
_THETA_THRESHOLD = 12.0 * np.pi / 180.0  # ~0.209 rad

# Lyapunov weights for cart-centering terms (small relative to pole terms)
_C_X = 0.1
_C_V = 0.01


class _CartPoleContinuousEnv(gym.Env):
    """
    CartPole with continuous force action ∈ [−10, 10].

    Uses identical physics to CartPole-v1 but replaces the discrete push
    {0→−F, 1→+F} with a scalar force in [−F, F].
    Rendering is done by syncing state into a CartPole-v1 instance.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        high = np.array(
            [_X_THRESHOLD * 2, np.finfo(np.float32).max,
             _THETA_THRESHOLD * 2, np.finfo(np.float32).max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-_FORCE_MAG, high=_FORCE_MAG, shape=(1,), dtype=np.float32
        )
        self._state: np.ndarray | None = None
        self._steps = 0
        self._max_steps = 500
        self._render_env = (
            gym.make("CartPole-v1", render_mode=render_mode)
            if render_mode is not None
            else None
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self.np_random.uniform(
            low=-0.05, high=0.05, size=(4,)
        ).astype(np.float32)
        self._steps = 0
        return self._state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self._state
        force = float(np.clip(np.asarray(action).flat[0], -_FORCE_MAG, _FORCE_MAG))

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        temp = (force + _POLEMASS_LENGTH * theta_dot ** 2 * sin_th) / _TOTAL_MASS
        theta_acc = (_GRAVITY * sin_th - cos_th * temp) / (
            _HALF_LENGTH * (4.0 / 3.0 - _MASSPOLE * cos_th ** 2 / _TOTAL_MASS)
        )
        x_acc = temp - _POLEMASS_LENGTH * theta_acc * cos_th / _TOTAL_MASS

        x = x + _DT * x_dot
        x_dot = x_dot + _DT * x_acc
        theta = theta + _DT * theta_dot
        theta_dot = theta_dot + _DT * theta_acc

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._steps += 1

        terminated = bool(
            x < -_X_THRESHOLD or x > _X_THRESHOLD
            or theta < -_THETA_THRESHOLD or theta > _THETA_THRESHOLD
        )
        truncated = self._steps >= self._max_steps
        return self._state.copy(), 1.0, terminated, truncated, {}

    def render(self):
        if self._render_env is not None:
            self._render_env.unwrapped.state = self._state.copy()
            return self._render_env.render()

    def close(self):
        if self._render_env is not None:
            self._render_env.close()
        super().close()


class LyapunovCartPole(LyapunovWrapper):
    """
    Continuous CartPole with Lyapunov reward R_t = Φ(s_t) - Φ(s_{t+1}).

    State: [x, ẋ, θ, θ̇]  (cart position/velocity, pole angle/angular velocity)

    Lyapunov function: sum of pole energy and soft cart-centering terms.
        Φ(s) = (1 − cosθ) + 0.5 θ̇² + c_x x² + c_v ẋ²
    Zero at the upright balanced equilibrium (θ=0, θ̇=0, x=0, ẋ=0).
    """

    obs_dim = 4
    act_dim = 1
    action_clip = (-_FORCE_MAG, _FORCE_MAG)
    k_action = 0.001

    def __init__(self, render_mode=None, native_reward: bool = False):
        super().__init__(
            _CartPoleContinuousEnv(render_mode=render_mode),
            native_reward=native_reward,
        )

    @staticmethod
    def phi(obs: np.ndarray) -> float:
        x, x_dot, theta, theta_dot = obs[0], obs[1], obs[2], obs[3]
        return float(
            (1.0 - np.cos(theta))
            + 0.5 * theta_dot ** 2
            + _C_X * x ** 2
            + _C_V * x_dot ** 2
        )

    @staticmethod
    def phi_terms(obs: np.ndarray) -> dict[str, float]:
        x, x_dot, theta, theta_dot = obs[0], obs[1], obs[2], obs[3]
        return {
            "pole_angle":    float(1.0 - np.cos(theta)),
            "pole_velocity": float(0.5 * theta_dot ** 2),
            "cart_position": float(_C_X * x ** 2),
            "cart_velocity": float(_C_V * x_dot ** 2),
        }

    @staticmethod
    def phi_torch(obs: torch.Tensor) -> torch.Tensor:
        x       = obs[..., 0:1]
        x_dot   = obs[..., 1:2]
        theta   = obs[..., 2:3]
        theta_dot = obs[..., 3:4]
        return (
            (1.0 - torch.cos(theta))
            + 0.5 * theta_dot ** 2
            + _C_X * x ** 2
            + _C_V * x_dot ** 2
        )

    @staticmethod
    def state_grid(n: int = 50):
        theta = np.linspace(-_THETA_THRESHOLD, _THETA_THRESHOLD, n)
        theta_dot = np.linspace(-3.0, 3.0, n)
        TH, TD = np.meshgrid(theta, theta_dot)
        # Slice at x=0, ẋ=0 for 2-D visualisation
        zeros = np.zeros_like(TH)
        states = np.stack([zeros, zeros, TH, TD], axis=-1).reshape(-1, 4).astype(np.float32)
        return TH, TD, states

    def shaping_reward(self, obs, action, next_obs, terminated) -> dict[str, float]:
        return {
            "action_penalty": -self.k_action * float(np.sum(action ** 2)),
        }
