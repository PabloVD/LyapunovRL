"""
evaluate.py — compute all 6 evaluation metrics and produce plots.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --env_id Pendulum-v1 --out_dir plots/
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from envs.ant import LyapunovAnt
from envs.base import LyapunovWrapper
from envs.cartpole import LyapunovCartPole
from envs.mountain_car import LyapunovMountainCar
from envs.pendulum import LyapunovPendulum
from models.actor import Actor
from models.critic import make_critic

ENV_REGISTRY: dict[str, type[LyapunovWrapper]] = {
    "Pendulum-v1": LyapunovPendulum,
    "MountainCarContinuous-v0": LyapunovMountainCar,
    "CartPoleContinuous-v0": LyapunovCartPole,
    "Ant-v4": LyapunovAnt,
}

COLORS = {"A": "#3269FF", "B": "#E01A1D"}
LABELS = {"A": "Condition A — standard critic", "B": "Condition B — Lyapunov structured"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── TensorBoard helpers ───────────────────────────────────────────────────────

def load_tb_scalar(run_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def collect_curves(runs_dir: str, tag: str, n_seeds: int = 5):
    results = {}
    for cond in ["A", "B"]:
        seed_curves = []
        for seed in range(n_seeds):
            run_dir = os.path.join(runs_dir, f"condition_{cond}_seed_{seed}")
            if not os.path.exists(run_dir):
                continue
            steps, values = load_tb_scalar(run_dir, tag)
            if len(values):
                seed_curves.append((steps, values))
        results[cond] = seed_curves
    return results


def align_curves(seed_curves, n_points=300):
    if not seed_curves:
        return None, None, None
    valid = [(s, v) for s, v in seed_curves if len(s) >= 2]
    if not valid:
        return None, None, None
    # Use the intersection of all seeds' step ranges so no seed is extrapolated.
    step_lo = max(s.min() for s, _ in valid)
    step_hi = min(s.max() for s, _ in valid)
    if step_lo >= step_hi:
        return None, None, None
    grid = np.linspace(step_lo, step_hi, n_points)
    interpolated = [np.interp(grid, steps, values) for steps, values in valid]
    arr = np.stack(interpolated)
    return grid, arr.mean(0), arr.std(0)



# ── Evaluation rollout ─────────────────────────────────────────────────────────

def eval_policy(actor: Actor, env: LyapunovWrapper, n_episodes: int = 10, device="cuda"):
    actor.eval()
    ep_rewards, upright_fracs = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r, upright, steps, done = 0.0, 0, 0, False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(obs_t).mean.squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, _ = env.step(np.clip(action, -2.0, 2.0))
            ep_r += r
            upright += int(abs(np.arctan2(obs[1], obs[0])) < 0.1)
            steps += 1
            done = terminated or truncated
        ep_rewards.append(ep_r)
        upright_fracs.append(upright / max(steps, 1))
    return np.array(ep_rewards), np.array(upright_fracs)


# ── Value grid ────────────────────────────────────────────────────────────────

def value_grid(critic, env_class: type[LyapunovWrapper], device, n=50):
    ax1, ax2, states = env_class.state_grid(n)
    if states is None:
        return None, None, None, None
    states_t = torch.tensor(states, device=device)
    with torch.no_grad():
        phi_vals = env_class.phi_torch(states_t).cpu().numpy().reshape(n, n)
        v_vals = critic(states_t).cpu().numpy().reshape(n, n)
    return ax1, ax2, phi_vals, v_vals


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_learning_curves(runs_dir: str, out_dir: str):
    curves = collect_curves(runs_dir, "train/ep_reward")
    fig, ax = plt.subplots(figsize=(9, 5))

    # Compute a shared normalisation scale from all conditions so the
    # y-axis is readable regardless of the Lyapunov function's magnitude.
    aligned = {c: align_curves(curves[c]) for c in ["A", "B"]}
    all_means = [m for _, m, _ in aligned.values() if m is not None]
    scale = max(np.abs(m).max() for m in all_means) if all_means else 1.0
    if scale == 0:
        scale = 1.0

    any_plotted = False
    for cond in ["A", "B"]:
        grid, mean, std = aligned[cond]
        if grid is None:
            continue
        ax.plot(grid, mean / scale, color=COLORS[cond], label=LABELS[cond], linewidth=2)
        ax.fill_between(grid, (mean - std) / scale, (mean + std) / scale,
                        alpha=0.15, color=COLORS[cond])
        any_plotted = True

    ax.set_xlabel("Environment steps", labelpad=8)
    ax.set_ylabel("Episode reward (normalised)", labelpad=8)
    ax.set_title("Learning Curves — mean ± std over 5 seeds", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.minorticks_on()
    if any_plotted:
        ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved learning_curves.png")


def plot_value_loss(runs_dir: str, out_dir: str):
    """Plot value loss (MSE of V vs GAE returns) over training steps."""
    curves = collect_curves(runs_dir, "loss/value")
    fig, ax = plt.subplots(figsize=(9, 5))

    aligned = {c: align_curves(curves[c]) for c in ["A", "B"]}
    all_means = [m for _, m, _ in aligned.values() if m is not None]
    scale = max(m.max() for m in all_means) if all_means else 1.0
    if scale == 0:
        scale = 1.0

    any_plotted = False
    for cond in ["A", "B"]:
        grid, mean, std = aligned[cond]
        if grid is None:
            continue
        ax.plot(grid, mean / scale, color=COLORS[cond], label=LABELS[cond], linewidth=2)
        ax.fill_between(grid, (mean - std) / scale, (mean + std) / scale,
                        alpha=0.15, color=COLORS[cond])
        any_plotted = True

    ax.set_xlabel("Environment steps", labelpad=8)
    ax.set_ylabel("Value loss (normalised)", labelpad=8)
    ax.set_title("Critic MSE over Training — mean ± std over 5 seeds", pad=12)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.minorticks_on()
    if any_plotted:
        ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "value_loss.png"), dpi=150)
    plt.close(fig)
    print("Saved value_loss.png")


AXIS_LABELS = {
    "Pendulum-v1": ("θ (rad)", "θ̇ (rad/s)"),
    "MountainCarContinuous-v0": ("x (position)", "v (velocity)"),
    "CartPoleContinuous-v0": ("θ (rad)", "θ̇ (rad/s)"),
}

def plot_value_heatmaps(run_name_A: str, run_name_B: str, runs_dir: str,
                        out_dir: str, env_class: type[LyapunovWrapper],
                        env_id: str, device="cuda"):
    obs_dim = env_class.obs_dim
    xlabel, ylabel = AXIS_LABELS.get(env_id, ("dim 0", "dim 1"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax_idx, (condition, run_name, title) in enumerate([
        ("A", run_name_A, "Condition A: V(s)"),
        ("B", run_name_B, "Condition B: V(s) = Φ + f"),
    ]):
        critic = make_critic(condition, obs_dim=obs_dim, hidden_dim=64,
                             phi_fn=env_class.phi_torch).to(device)
        ckpt_path = os.path.join(runs_dir, run_name, "critic.pt")
        if os.path.exists(ckpt_path):
            critic.load_state_dict(torch.load(ckpt_path, map_location=device))
        ax1, ax2, phi_vals, v_vals = value_grid(critic, env_class, device)
        im = axes[ax_idx].pcolormesh(ax1, ax2, v_vals, cmap="viridis")
        axes[ax_idx].set_title(title)
        axes[ax_idx].set_xlabel(xlabel)
        axes[ax_idx].set_ylabel(ylabel)
        fig.colorbar(im, ax=axes[ax_idx])

    im = axes[2].pcolormesh(ax1, ax2, phi_vals, cmap="viridis")
    axes[2].set_title("Analytical Φ(s)")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "value_heatmaps.png"), dpi=150)
    plt.close(fig)
    print("Saved value_heatmaps.png")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--env_id", type=str, default="Pendulum-v1", choices=list(ENV_REGISTRY))
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment subfolder created by run_experiments.py")
    parser.add_argument("--out_dir", type=str, default="plots")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    env_class = ENV_REGISTRY[args.env_id]
    env_runs_dir = os.path.join(args.runs_dir, args.env_id)
    if args.exp_id:
        env_runs_dir = os.path.join(env_runs_dir, args.exp_id)
    out_dir = os.path.join(args.out_dir, args.env_id)
    if args.exp_id:
        out_dir = os.path.join(out_dir, args.exp_id)
    os.makedirs(out_dir, exist_ok=True)

    plot_learning_curves(env_runs_dir, out_dir)
    plot_value_loss(env_runs_dir, out_dir)

    run_A = "condition_A_seed_0"
    run_B = "condition_B_seed_0"
    grid_supported = env_class.state_grid()[0] is not None
    if not grid_supported:
        print(f"Skipping value heatmaps ({args.env_id} does not support state_grid visualisation).")
    elif os.path.exists(os.path.join(env_runs_dir, run_A, "critic.pt")) and \
         os.path.exists(os.path.join(env_runs_dir, run_B, "critic.pt")):
        plot_value_heatmaps(run_A, run_B, env_runs_dir, out_dir, env_class, args.env_id, args.device)
    else:
        print("Skipping value heatmaps (no checkpoints found — train first).")


if __name__ == "__main__":
    main()
