"""
train.py — train one PPO run for a given condition and seed.

Usage:
    uv run python train.py --condition A --seed 0
    uv run python train.py --condition B --seed 0 --total_steps 300000
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from envs.ant import LyapunovAnt
from envs.base import LyapunovWrapper
from envs.cartpole import LyapunovCartPole
from envs.pendulum import LyapunovPendulum
from envs.mountain_car import LyapunovMountainCar
from ppo.trainer import PPOTrainer

ENV_REGISTRY: dict[str, type[LyapunovWrapper]] = {
    "Pendulum-v1": LyapunovPendulum,
    "MountainCarContinuous-v0": LyapunovMountainCar,
    "CartPoleContinuous-v0": LyapunovCartPole,
    "Ant-v4": LyapunovAnt,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: Config):
    if not cfg.run_name:
        cfg.run_name = f"condition_{cfg.condition}_seed_{cfg.seed}"

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run_dir = f"runs/{cfg.env_id}/{cfg.run_name}"
    cfg.save(f"{run_dir}/config.yaml")
    writer = SummaryWriter(log_dir=run_dir)
    env_class = ENV_REGISTRY[cfg.env_id]

    # Apply target velocity to Ant before creating the env.
    if hasattr(env_class, "_target_vx"):
        env_class._target_vx = cfg.target_velocity

    env = env_class(native_reward=cfg.native_reward)

    # Wrap with running-stats observation normalisation when requested.
    # NormalizeObservation is applied OUTSIDE LyapunovWrapper so that reward
    # computation (Φ(s_t) − Φ(s_{t+1})) always sees raw observations.
    norm_env = None
    if cfg.normalize_obs:
        import gymnasium
        norm_env = gymnasium.wrappers.NormalizeObservation(env)
        env = norm_env

    trainer = PPOTrainer(cfg, device, writer,
                         obs_dim=env_class.obs_dim, act_dim=env_class.act_dim,
                         phi_fn=env_class.phi_torch,
                         action_cost_alpha=getattr(env_class, "k_action", 0.0))

    obs, _ = env.reset(seed=cfg.seed)
    n_updates = cfg.total_steps // cfg.T
    ep_rewards = []
    ep_reward = 0.0
    ep_len = 0
    ep_reward_terms: dict[str, float] = {}
    ep_step_rewards: list[float] = []

    pbar = tqdm(range(1, n_updates + 1), desc=f"[{cfg.condition}|seed{cfg.seed}]", unit="update")

    for update in pbar:
        trainer.global_step = (update - 1) * cfg.T

        act_lo, act_hi = env_class.action_clip
        for _ in range(cfg.T):
            action, log_prob, value = trainer.get_action_and_value(obs)
            clipped_action = np.clip(action, act_lo, act_hi)

            next_obs, reward, terminated, truncated, info = env.step(clipped_action)

            # Compute the correct next-state value before potentially resetting.
            # - terminated: absorbing state, V(next) = 0.
            # - truncated:  timeout, episode continues hypothetically — use V(next_obs).
            # - otherwise:  filled from the buffer in compute_gae.
            if terminated:
                next_value = 0.0
            elif truncated:
                with torch.no_grad():
                    nobs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    next_value = trainer.critic(nobs_t).squeeze().item()
            else:
                next_value = 0.0  # placeholder; overwritten by compute_gae

            trainer.buffer.add(obs, action, reward, terminated, truncated, log_prob, value, next_value)

            ep_reward += reward
            ep_len += 1
            ep_step_rewards.append(float(reward))
            for k, v in info.get("reward_terms", {}).items():
                ep_reward_terms[k] = ep_reward_terms.get(k, 0.0) + v
            obs = next_obs

            if terminated or truncated:
                ep_rewards.append(ep_reward)
                writer.add_scalar("train/ep_reward", ep_reward, trainer.global_step)
                writer.add_scalar("train/ep_len", ep_len, trainer.global_step)
                for k, v in ep_reward_terms.items():
                    writer.add_scalar(f"train/ep_{k}", v, trainer.global_step)
                if len(ep_step_rewards) > 1:
                    r_arr = np.array(ep_step_rewards)
                    discounts = cfg.gamma ** np.arange(len(r_arr))
                    dr_arr = discounts * r_arr
                    snr = np.abs(dr_arr.mean()) / (dr_arr.std() + 1e-8)
                    writer.add_scalar("train/ep_reward_snr", snr, trainer.global_step)
                ep_reward = 0.0
                ep_len = 0
                ep_reward_terms = {}
                ep_step_rewards = []
                obs, _ = env.reset()

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = trainer.critic(obs_t).squeeze().item()

        # Sync normaliser stats into the env class so phi_torch can denormalise
        # when condition B is used with normalize_obs.
        if norm_env is not None and hasattr(env_class, "obs_rms_mean"):
            env_class.obs_rms_mean = norm_env.obs_rms.mean.copy()
            env_class.obs_rms_std = np.sqrt(norm_env.obs_rms.var + 1e-8)

        trainer.update(last_value)

        mean_r = np.mean(ep_rewards[-10:]) if ep_rewards else float("nan")
        pbar.set_postfix(steps=trainer.global_step, mean_r=f"{mean_r:.3f}")

    torch.save(trainer.actor.state_dict(), f"{run_dir}/actor.pt")
    torch.save(trainer.critic.state_dict(), f"{run_dir}/critic.pt")

    writer.close()
    env.close()
    print(f"Done. Checkpoints saved to {run_dir}/")
    return trainer


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file. CLI flags override YAML values.")
    parser.add_argument("--condition", type=str, default=None, choices=["A", "B", "C"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--env_id", type=str, default=None, choices=list(ENV_REGISTRY))
    parser.add_argument("--native_reward", action="store_true", default=None,
                        help="Use the env's original reward instead of Lyapunov shaping.")
    parser.add_argument("--normalize_obs", action="store_true", default=None,
                        help="Wrap env with running-stats NormalizeObservation (recommended for MuJoCo).")
    parser.add_argument("--target_velocity", type=float, default=None,
                        help="Target forward speed for Ant-v4 Lyapunov function (m/s).")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()

    overrides = {k: v for k, v in vars(args).items()
                 if k != "config" and v is not None}
    for k, v in overrides.items():
        setattr(cfg, k, v)

    return cfg


if __name__ == "__main__":
    train(parse_args())
