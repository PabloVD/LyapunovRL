"""
visualize.py — render the learned policy for a trained run.

Live display (opens a window):
    uv run python visualize.py --condition A --seed 0
    uv run python visualize.py --condition B --seed 0

Save as GIF:
    uv run python visualize.py --condition A --seed 0 --save
    uv run python visualize.py --condition B --seed 0 --save --out_dir plots/
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from envs.base import LyapunovWrapper
from envs.mountain_car import LyapunovMountainCar
from envs.pendulum import LyapunovPendulum
from models.actor import Actor

ENV_REGISTRY: dict[str, type[LyapunovWrapper]] = {
    "Pendulum-v1": LyapunovPendulum,
    "MountainCarContinuous-v0": LyapunovMountainCar,
}


def load_actor(run_dir: str, obs_dim: int, act_dim: int,
               hidden_dim: int, device: str) -> Actor:
    ckpt = os.path.join(run_dir, "actor.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt} — train first.")
    actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
    actor.load_state_dict(torch.load(ckpt, map_location=device))
    actor.eval()
    return actor


def run_episode(actor: Actor, env: LyapunovWrapper,
                device: str, collect_frames: bool) -> tuple[float, list]:
    obs, _ = env.reset()
    total_reward = 0.0
    frames = []
    done = False

    while not done:
        if collect_frames:
            frames.append(env.render())

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = actor(obs_t).mean.squeeze(0).cpu().numpy()

        act_lo, act_hi = type(env).action_clip
        obs, r, terminated, truncated, _ = env.step(np.clip(action, act_lo, act_hi))
        total_reward += r
        done = terminated or truncated

    return total_reward, frames


def save_gif(frames: list[np.ndarray], path: str, fps: int = 30):
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved GIF → {path}  ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_id", type=str, default="Pendulum-v1",
                        choices=list(ENV_REGISTRY))
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment subfolder created by run_experiments.py")
    parser.add_argument("--save", action="store_true",
                        help="Save a GIF instead of opening a live window")
    parser.add_argument("--out_dir", type=str, default="plots")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--native_reward", action="store_true",
                        help="Visualize a policy trained with --native_reward.")
    args = parser.parse_args()

    env_class = ENV_REGISTRY[args.env_id]
    exp_part = [args.exp_id] if args.exp_id else []
    run_dir = os.path.join(args.runs_dir, args.env_id, *exp_part,
                           f"condition_{args.condition}_seed_{args.seed}")
    device = args.device if torch.cuda.is_available() else "cpu"

    actor = load_actor(run_dir, obs_dim=env_class.obs_dim, act_dim=env_class.act_dim,
                       hidden_dim=args.hidden_dim, device=device)

    render_mode = "rgb_array" if args.save else "human"
    env = env_class(render_mode=render_mode, native_reward=args.native_reward)

    all_frames = []
    for ep in range(args.n_episodes):
        reward, frames = run_episode(actor, env, device, collect_frames=args.save)
        all_frames.extend(frames)
        print(f"  Episode {ep + 1}/{args.n_episodes}  total reward: {reward:.3f}")

    env.close()

    if args.save:
        out_dir = os.path.join(args.out_dir, args.env_id)
        os.makedirs(out_dir, exist_ok=True)
        gif_path = os.path.join(
            out_dir, f"policy_cond{args.condition}_seed{args.seed}.gif"
        )
        save_gif(all_frames, gif_path)


if __name__ == "__main__":
    main()
