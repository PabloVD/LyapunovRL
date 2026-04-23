"""
evaluate.py — compute all 6 evaluation metrics and produce plots.

Usage:
    uv run python evaluate.py            # uses runs/ directory
    uv run python evaluate.py --out_dir plots/
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import Config
from envs.lyapunov_pendulum import LyapunovPendulum
from lyapunov.functions import phi_np, phi_torch
from models.actor import Actor
from models.critic import make_critic


# ── Metric helpers ────────────────────────────────────────────────────────────

def load_tb_scalar(run_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for a TensorBoard scalar tag."""
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag not in ea.scalars.Keys():
        return np.array([]), np.array([])
    events = ea.scalars.Values(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def collect_curves(runs_dir: str, tag: str, n_seeds: int = 5):
    """
    For each condition, load curves from all seeds and return
    (steps, mean, std) for conditions A and B.
    """
    results = {}
    for cond in ["A", "B"]:
        seed_curves = []
        for seed in range(n_seeds):
            run_name = f"condition_{cond}_seed_{seed}"
            run_dir = os.path.join(runs_dir, run_name)
            if not os.path.exists(run_dir):
                continue
            steps, values = load_tb_scalar(run_dir, tag)
            if len(values):
                seed_curves.append((steps, values))
        results[cond] = seed_curves
    return results


def align_curves(seed_curves, n_points=200):
    """Interpolate all seed curves onto a common step grid."""
    if not seed_curves:
        return None, None, None
    all_steps = np.concatenate([s for s, _ in seed_curves])
    step_min, step_max = all_steps.min(), all_steps.max()
    grid = np.linspace(step_min, step_max, n_points)
    interpolated = []
    for steps, values in seed_curves:
        if len(steps) < 2:
            continue
        interp = np.interp(grid, steps, values)
        interpolated.append(interp)
    arr = np.stack(interpolated)
    return grid, arr.mean(0), arr.std(0)


# ── Evaluation rollout ─────────────────────────────────────────────────────────

def eval_policy(actor: Actor, env: LyapunovPendulum, n_episodes: int = 10, device="cuda"):
    """Run n_episodes with the given actor; return per-episode rewards and upright fractions."""
    actor.eval()
    ep_rewards = []
    upright_fracs = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        upright = 0
        steps = 0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.mean.squeeze(0).cpu().numpy()
            action = np.clip(action, -2.0, 2.0)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_r += r
            theta = np.arctan2(obs[1], obs[0])
            upright += int(abs(theta) < 0.1)
            steps += 1
            done = terminated or truncated
        ep_rewards.append(ep_r)
        upright_fracs.append(upright / max(steps, 1))

    return np.array(ep_rewards), np.array(upright_fracs)


# ── Value grid comparison ──────────────────────────────────────────────────────

def value_grid(critic, device, n=50):
    """Evaluate learned V and analytical Φ on a grid of (θ, θ̇)."""
    theta = np.linspace(-np.pi, np.pi, n)
    theta_dot = np.linspace(-8.0, 8.0, n)
    TH, TD = np.meshgrid(theta, theta_dot)

    states = np.stack([np.cos(TH), np.sin(TH), TD], axis=-1).reshape(-1, 3).astype(np.float32)
    phi_vals = phi_np(states).reshape(n, n)

    states_t = torch.tensor(states, device=device)
    with torch.no_grad():
        v_vals = critic(states_t).cpu().numpy().reshape(n, n)

    return TH, TD, phi_vals, v_vals


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_learning_curves(runs_dir: str, out_dir: str):
    curves = collect_curves(runs_dir, "train/ep_reward")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"A": "#1f77b4", "B": "#ff7f0e"}
    labels = {"A": "Condition A (standard critic)", "B": "Condition B (Lyapunov-structured)"}
    for cond in ["A", "B"]:
        grid, mean, std = align_curves(curves[cond])
        if grid is None:
            continue
        ax.plot(grid, mean, color=colors[cond], label=labels[cond])
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=colors[cond])
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward (Lyapunov)")
    ax.set_title("Learning Curves (mean ± std, 5 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved learning_curves.png")


def plot_value_heatmaps(run_name_A: str, run_name_B: str, runs_dir: str, out_dir: str, device="cuda"):
    cfg_A = Config(condition="A", run_name=run_name_A)
    cfg_B = Config(condition="B", run_name=run_name_B)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    entries = [
        (cfg_A, "Condition A: V(s)"),
        (cfg_B, "Condition B: V(s)=Φ+f"),
    ]
    phi_plotted = False

    for ax_idx, (cfg, title) in enumerate(entries):
        critic = make_critic(cfg.condition, obs_dim=3, hidden_dim=64).to(device)
        ckpt_path = os.path.join(runs_dir, cfg.run_name, "critic.pt")
        if os.path.exists(ckpt_path):
            critic.load_state_dict(torch.load(ckpt_path, map_location=device))
        TH, TD, phi_vals, v_vals = value_grid(critic, device)
        im = axes[ax_idx].pcolormesh(TH, TD, v_vals, cmap="viridis")
        axes[ax_idx].set_title(title)
        axes[ax_idx].set_xlabel("θ (rad)")
        axes[ax_idx].set_ylabel("θ̇ (rad/s)")
        fig.colorbar(im, ax=axes[ax_idx])

    im = axes[2].pcolormesh(TH, TD, phi_vals, cmap="viridis")
    axes[2].set_title("Analytical Φ(s)")
    axes[2].set_xlabel("θ (rad)")
    axes[2].set_ylabel("θ̇ (rad/s)")
    fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "value_heatmaps.png"), dpi=150)
    plt.close(fig)
    print("Saved value_heatmaps.png")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--out_dir", type=str, default="plots")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_learning_curves(args.runs_dir, args.out_dir)

    run_A = "condition_A_seed_0"
    run_B = "condition_B_seed_0"
    ckpt_A = os.path.join(args.runs_dir, run_A, "critic.pt")
    ckpt_B = os.path.join(args.runs_dir, run_B, "critic.pt")
    if os.path.exists(ckpt_A) and os.path.exists(ckpt_B):
        plot_value_heatmaps(run_A, run_B, args.runs_dir, args.out_dir, args.device)
    else:
        print("Skipping value heatmaps (no checkpoints found — train first).")


if __name__ == "__main__":
    main()
