"""
train.py — train one PPO run for a given condition and seed.

Usage:
    uv run python train.py --condition A --seed 0
    uv run python train.py --condition B --seed 0 --total_steps 300000
"""

import argparse
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from envs.lyapunov_pendulum import LyapunovPendulum
from ppo.trainer import PPOTrainer


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

    writer = SummaryWriter(log_dir=f"runs/{cfg.run_name}")
    env = LyapunovPendulum()
    trainer = PPOTrainer(cfg, device, writer)

    obs, _ = env.reset(seed=cfg.seed)
    n_updates = cfg.total_steps // cfg.T
    ep_rewards = []
    ep_reward = 0.0
    ep_len = 0

    print(f"Training condition={cfg.condition} seed={cfg.seed} device={device} "
          f"total_steps={cfg.total_steps}")

    start = time.time()

    for update in range(1, n_updates + 1):
        trainer.global_step = (update - 1) * cfg.T

        for _ in range(cfg.T):
            action, log_prob, value = trainer.get_action_and_value(obs)
            clipped_action = np.clip(action, -2.0, 2.0)

            next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated

            trainer.buffer.add(obs, action, reward, float(done), log_prob, value)

            ep_reward += reward
            ep_len += 1
            obs = next_obs

            if done:
                ep_rewards.append(ep_reward)
                writer.add_scalar("train/ep_reward", ep_reward, trainer.global_step)
                writer.add_scalar("train/ep_len", ep_len, trainer.global_step)
                ep_reward = 0.0
                ep_len = 0
                obs, _ = env.reset()

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = trainer.critic(obs_t).squeeze().item()

        trainer.update(last_value)

        if update % 10 == 0:
            mean_r = np.mean(ep_rewards[-10:]) if ep_rewards else float("nan")
            elapsed = time.time() - start
            print(f"  update {update:4d}/{n_updates} | "
                  f"steps {trainer.global_step:7d} | "
                  f"mean_ep_reward {mean_r:7.3f} | "
                  f"elapsed {elapsed:.1f}s")

    # Save checkpoints for evaluation
    ckpt_dir = f"runs/{cfg.run_name}"
    torch.save(trainer.actor.state_dict(), f"{ckpt_dir}/actor.pt")
    torch.save(trainer.critic.state_dict(), f"{ckpt_dir}/critic.pt")

    writer.close()
    env.close()
    print(f"Done. Checkpoints saved to {ckpt_dir}/")
    return trainer


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()
    cfg = Config(
        condition=args.condition,
        seed=args.seed,
        total_steps=args.total_steps,
        gamma=args.gamma,
        device=args.device,
        run_name=args.run_name,
    )
    return cfg


if __name__ == "__main__":
    train(parse_args())
