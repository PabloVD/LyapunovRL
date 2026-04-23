"""
run_experiments.py — launch all 10 runs (2 conditions × 5 seeds) sequentially.

Usage:
    uv run python run_experiments.py
    uv run python run_experiments.py --total_steps 300000 --device cuda
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    runs = [
        {"condition": cond, "seed": seed}
        for cond in ["A", "B"]
        for seed in range(1)
    ]

    print(f"Launching {len(runs)} runs ({args.total_steps} steps each)...")
    for i, run in enumerate(runs, 1):
        cond = run["condition"]
        seed = run["seed"]
        run_name = f"condition_{cond}_seed_{seed}"
        print(f"\n[{i}/{len(runs)}] {run_name}")
        cmd = [
            sys.executable, "train.py",
            "--condition", cond,
            "--seed", str(seed),
            "--total_steps", str(args.total_steps),
            "--device", args.device,
            "--run_name", run_name,
        ]
        subprocess.run(cmd, check=True)

    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
