"""
run_experiments.py — launch all runs (2 conditions × N seeds) sequentially.

Usage:
    uv run python run_experiments.py --config configs/ant.yaml
    uv run python run_experiments.py --config configs/ant.yaml --num_seeds 3
    uv run python run_experiments.py --config configs/ant.yaml --total_steps 1000000
    uv run python run_experiments.py --env_id Pendulum-v1   # no YAML, uses defaults
    uv run python run_experiments.py --condition B          # only condition B
"""

import argparse
import subprocess
import sys
from datetime import datetime

from config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a Config YAML. CLI flags override YAML values.")
    # Experiment-level args (not forwarded to train.py, used here only).
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--exp_id", type=str, default=None,
                        help="Experiment folder name. Defaults to a timestamp.")
    parser.add_argument("--condition", type=str, default=None, choices=["A", "B"],
                        help="Run only this condition. Defaults to both A and B.")
    # Per-field CLI overrides — all optional, take precedence over YAML when given.
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--env_id", type=str, default=None)
    parser.add_argument("--native_reward", action="store_true", default=None,
                        help="Use env's original reward. Only runs condition A.")
    parser.add_argument("--normalize_obs", action="store_true", default=None,
                        help="Wrap env with running-stats NormalizeObservation.")
    parser.add_argument("--target_velocity", type=float, default=None,
                        help="Target forward speed for Ant-v4 (m/s).")
    args = parser.parse_args()

    # Build effective config: YAML base → CLI overrides.
    cfg = Config.from_yaml(args.config) if args.config else Config()

    # Track which fields were overridden on the CLI so we can re-pass them to
    # train.py (which also loads the YAML, so explicit overrides must win).
    cli_overrides: list[str] = []

    for attr, value, flag in [
        ("total_steps", args.total_steps, "--total_steps"),
        ("device", args.device, "--device"),
        ("env_id", args.env_id, "--env_id"),
        ("target_velocity", args.target_velocity, "--target_velocity"),
    ]:
        if value is not None:
            setattr(cfg, attr, value)
            cli_overrides.extend([flag, str(value)])

    for attr, value, flag in [
        ("native_reward", args.native_reward, "--native_reward"),
        ("normalize_obs", args.normalize_obs, "--normalize_obs"),
    ]:
        if value:
            setattr(cfg, attr, True)
            cli_overrides.append(flag)

    # Determine which conditions to run.
    if cfg.native_reward:
        conditions = ["A"]
    elif args.condition:
        conditions = [args.condition]
    else:
        conditions = ["A", "B"]

    exp_id = args.exp_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs = [
        {"condition": cond, "seed": seed}
        for cond in conditions
        for seed in range(args.num_seeds)
    ]

    print(f"Experiment : {exp_id}")
    print(f"Config     : {args.config or '(defaults)'}")
    print(f"Env        : {cfg.env_id}  |  steps: {cfg.total_steps:,}  |  {len(runs)} runs")

    for i, run in enumerate(runs, 1):
        cond, seed = run["condition"], run["seed"]
        run_name = f"{exp_id}/condition_{cond}_seed_{seed}"
        print(f"\n[{i}/{len(runs)}] {run_name}")
        cmd = [
            sys.executable, "train.py",
            "--condition", cond,
            "--seed", str(seed),
            "--run_name", run_name,
        ]
        if args.config:
            cmd.extend(["--config", args.config])
        cmd.extend(cli_overrides)
        subprocess.run(cmd, check=True)

    print("\nAll runs complete. Running evaluation...")
    subprocess.run([
        sys.executable, "evaluate.py",
        "--env_id", cfg.env_id,
        "--exp_id", exp_id,
        "--device", cfg.device,
    ], check=True)


if __name__ == "__main__":
    main()
