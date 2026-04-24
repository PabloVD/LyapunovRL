"""
run_experiments.py — launch all 10 runs (2 conditions × 5 seeds) sequentially.

Usage:
    uv run python run_experiments.py
    uv run python run_experiments.py --env_id MountainCarContinuous-v0
    uv run python run_experiments.py --total_steps 300000 --device cuda
    uv run python run_experiments.py --exp_id my_exp   # custom experiment name
"""

import argparse
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env_id", type=str, default="Pendulum-v1")
    parser.add_argument("--exp_id", type=str, default=None,
                        help="Experiment folder name. Defaults to a timestamp.")
    args = parser.parse_args()

    exp_id = args.exp_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    runs = [
        {"condition": cond, "seed": seed}
        for cond in ["A", "B"]
        for seed in range(5)
    ]

    print(f"Experiment: {exp_id}")
    print(f"Launching {len(runs)} runs ({args.total_steps} steps each)...")
    for i, run in enumerate(runs, 1):
        cond = run["condition"]
        seed = run["seed"]
        run_name = f"{exp_id}/condition_{cond}_seed_{seed}"
        print(f"\n[{i}/{len(runs)}] {run_name}")
        cmd = [
            sys.executable, "train.py",
            "--condition", cond,
            "--seed", str(seed),
            "--total_steps", str(args.total_steps),
            "--device", args.device,
            "--env_id", args.env_id,
            "--run_name", run_name,
        ]
        subprocess.run(cmd, check=True)

    print(f"\nAll runs complete. Evaluate with:")
    print(f"  uv run python evaluate.py --env_id {args.env_id} --exp_id {exp_id}")


if __name__ == "__main__":
    main()
