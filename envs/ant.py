"""
LyapunovAnt — Ant-v4 wrapped with a velocity-tracking Lyapunov function.

Observation layout (27D, exclude_current_positions_from_observation=True, no contacts):
  obs = concat(qpos[2:], qvel)
  obs[0]     = z (body height, ~0.75 when standing)
  obs[1:5]   = quaternion [w, x, y, z]
  obs[5:13]  = 8 joint angles
  obs[13]    = vx (forward velocity)
  obs[14]    = vy (lateral velocity)
  obs[15]    = vz (vertical velocity)
  obs[16:19] = body angular velocities [wx, wy, wz]
  obs[19:27] = 8 joint angular velocities

Lyapunov function:
  Φ(s) = C_VX·(vx − v*)²  +  C_VY·vy²  +  C_Z·max(0, z_ref − z)²

  Zero when the ant moves forward at v* without lateral drift and stays upright.
  R_t = Φ(s_t) − Φ(s_{t+1}) is positive when closing toward that regime.
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym

from envs.base import LyapunovWrapper

_Z_IDX = 0    # body height index
_VX_IDX = 13  # forward velocity index
_VY_IDX = 14  # lateral velocity index

_Z_REF = 0.3  # minimum viable body height (m)

_C_VX = 1.0   # velocity-tracking coefficient
_C_VY = 0.5   # lateral-drift penalty coefficient
_C_Z = 5.0    # height-deficit coefficient

_ACTION_PENALTY = 0.005  # per unit of squared action norm


class LyapunovAnt(LyapunovWrapper):
    """Ant-v4 with a velocity-tracking Lyapunov reward shaper.

    Class-level attributes obs_rms_mean / obs_rms_std are synced from the
    NormalizeObservation wrapper in train.py so that phi_torch can denormalize
    before computing Φ when condition B is used.
    """

    obs_dim = 27
    act_dim = 8
    action_clip = (-1.0, 1.0)
    k_action = _ACTION_PENALTY

    # Class-level: set from train.py after each rollout when normalize_obs=True.
    obs_rms_mean: np.ndarray | None = None
    obs_rms_std: np.ndarray | None = None

    # Target forward velocity — overridable via env_class._target_vx = cfg.target_velocity.
    _target_vx: float = 1.0

    def __init__(self, native_reward: bool = False):
        env = gym.make("Ant-v4")
        super().__init__(env, native_reward=native_reward)

    # ── Lyapunov function ──────────────────────────────────────────────────────

    @classmethod
    def phi(cls, obs: np.ndarray) -> float:
        z = float(obs[_Z_IDX])
        vx = float(obs[_VX_IDX])
        vy = float(obs[_VY_IDX])
        return (
            _C_VX * (vx - cls._target_vx) ** 2
            + _C_VY * vy ** 2
            + _C_Z * max(0.0, _Z_REF - z) ** 2
        )

    @classmethod
    def phi_torch(cls, obs: torch.Tensor) -> torch.Tensor:
        # Denormalize if running-stats are available (condition B with normalize_obs).
        if cls.obs_rms_mean is not None:
            mean = torch.tensor(cls.obs_rms_mean, device=obs.device, dtype=obs.dtype)
            std = torch.tensor(cls.obs_rms_std, device=obs.device, dtype=obs.dtype)
            obs = obs * std + mean
        z = obs[..., _Z_IDX : _Z_IDX + 1]
        vx = obs[..., _VX_IDX : _VX_IDX + 1]
        vy = obs[..., _VY_IDX : _VY_IDX + 1]
        return (
            _C_VX * (vx - cls._target_vx) ** 2
            + _C_VY * vy ** 2
            + _C_Z * torch.clamp(_Z_REF - z, min=0.0) ** 2
        )

    @classmethod
    def phi_terms(cls, obs: np.ndarray) -> dict[str, float]:
        z = float(obs[_Z_IDX])
        vx = float(obs[_VX_IDX])
        vy = float(obs[_VY_IDX])
        return {
            "velocity_tracking": _C_VX * (vx - cls._target_vx) ** 2,
            "lateral_velocity": _C_VY * vy ** 2,
            "height_deficit": _C_Z * max(0.0, _Z_REF - z) ** 2,
        }

    def shaping_reward(
        self, obs: np.ndarray, action: np.ndarray,
        next_obs: np.ndarray, terminated: bool,
    ) -> dict[str, float]:
        return {"action_penalty": -_ACTION_PENALTY * float(np.sum(action ** 2))}

    # ── Visualization (not supported for 27-D state space) ────────────────────

    @staticmethod
    def state_grid(n: int = 50):
        return (None, None, None)
