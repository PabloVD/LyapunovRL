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
from envs.base import LyapunovWrapper
from envs.cartpole import LyapunovCartPole
from envs.pendulum import LyapunovPendulum
from envs.mountain_car import LyapunovMountainCar
from ppo.trainer import PPOTrainer

ENV_REGISTRY: dict[str, type[LyapunovWrapper]] = {
    "Pendulum-v1": LyapunovPendulum,
    "MountainCarContinuous-v0": LyapunovMountainCar,
    "CartPoleContinuous-v0": LyapunovCartPole,
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
    writer = SummaryWriter(log_dir=run_dir)
    env_class = ENV_REGISTRY[cfg.env_id]
    env = env_class(native_reward=cfg.native_reward)
    trainer = PPOTrainer(cfg, device, writer,
                         obs_dim=env_class.obs_dim, act_dim=env_class.act_dim,
                         phi_fn=env_class.phi_torch)

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
                    snr = dr_arr.mean() / (dr_arr.std() + 1e-8)
                    writer.add_scalar("train/ep_reward_snr", snr, trainer.global_step)
                ep_reward = 0.0
                ep_len = 0
                ep_reward_terms = {}
                ep_step_rewards = []
                obs, _ = env.reset()

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = trainer.critic(obs_t).squeeze().item()

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
    parser.add_argument("--condition", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env_id", type=str, default="Pendulum-v1",
                        choices=list(ENV_REGISTRY))
    parser.add_argument("--native_reward", action="store_true",
                        help="Use the env's original reward instead of Lyapunov shaping.")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()
    cfg = Config(
        condition=args.condition,
        seed=args.seed,
        total_steps=args.total_steps,
        gamma=args.gamma,
        device=args.device,
        env_id=args.env_id,
        native_reward=args.native_reward,
        run_name=args.run_name,
    )
    return cfg


if __name__ == "__main__":
    train(parse_args())
