# Lyapunov-Based Value function for Reinforcement Learning

## Overview

This project investigates Lyapunov-based reward shaping in reinforcement learning, where the reward at each timestep is defined as the decrease in an analytically given Lyapunov function:

    R_t = Φ(s_t) - Φ(s_{t+1})

The core research question is whether structuring the value function as V(s) = Φ(s) + f(s) — where only the residual f(s) is learned — improves sample efficiency, stability, and policy quality compared to learning V(s) directly from scratch.

## Environment

**Gymnasium Pendulum-v1** — a standard continuous-control benchmark.

- **State space:** s = (cos θ, sin θ, θ̇) ∈ [-1,1] × [-1,1] × [-8,8]
- **Action space:** torque u ∈ [-2, 2] (continuous)
- **Goal:** stabilize the pendulum at the upright position (θ = 0)
- **Episode length:** 200 steps

The native Gymnasium reward is replaced entirely by the Lyapunov-based reward defined below.

We will expand to other envs afterwards.

## Lyapunov Function

The Lyapunov function is the **mechanical energy of the pendulum** relative to the upright equilibrium:

    Φ(s) = (1 - cos θ) + 0.5 * θ̇²

Where:
- θ = angle from upright (recovered from state as θ = atan2(sin θ, cos θ))
- θ̇ = angular velocity (third component of state)
- Φ(s) = 0 at the upright equilibrium (θ=0, θ̇=0)
- Φ(s) > 0 everywhere else (positive definite)

This is a valid Lyapunov candidate for the uncontrolled pendulum dynamics. The reward R_t = Φ(s_t) - Φ(s_{t+1}) is positive when the agent drives the system toward the equilibrium.

## Research Conditions

Two experimental conditions are compared using e.g. PPO:

### Condition A: Direct Value Learning
- Standard PPO with the Lyapunov reward R_t = Φ(s_t) - Φ(s_{t+1})
- Value function V(s) learned entirely from data (standard neural network critic)
- Baseline: represents the standard approach to critic learning

### Condition B: Structured Value Function
- Same PPO algorithm and Lyapunov reward
- Value function decomposed as: V(s) = Φ(s) + f_θ(s)
- Only the residual f_θ(s) is learned by the neural network
- Critic network outputs f_θ(s); total value = Φ(s) + f_θ(s)
- Hypothesis: initializing V with the physically meaningful Φ should improve sample efficiency and stability

## Usage

Install dependencies (requires [uv](https://github.com/astral-sh/uv)):

```bash
uv sync
```

Run the full experiment (2 conditions × 5 seeds):

```bash
uv run python run_experiments.py
```

Evaluate results and generate plots:

```bash
uv run python evaluate.py
```

View training curves in TensorBoard:

```bash
uv run tensorboard --logdir runs/
```

To train a single run manually:

```bash
uv run python train.py --condition A --seed 0
uv run python train.py --condition B --seed 0
```

To visualize a trained policy in a GUI:

```bash
uv run python visualize.py --condition A --seed 0
```

Or save as GIF (no display needed, e.g. on a server):

```bash
uv run python visualize.py --condition B --seed 0 --save
```

## Implementation Notes

- Custom PyTorch PPO with configurable critic (no stable-baselines3)
- Reward wrapper overrides native Gymnasium reward with R_t = Φ(s_t) - Φ(s_{t+1})
- For Condition B, only the residual f_θ(s) is learned; last layer zero-initialized so V ≈ Φ at training start
- Training: 300,000 environment steps per run, 5 random seeds each condition
- Hardware: GPU (NVIDIA RTX PRO 6000, `device='cuda'`)

## Evaluation Metrics

1. **Learning curves**: cumulative Lyapunov reward vs. environment steps (5 seeds, mean ± std)
2. **Sample efficiency**: steps to reach 90% of maximum average reward
3. **Upright stability**: fraction of time steps in upright region (|θ| < 0.1 rad) after training
4. **Value function quality**: compare learned V(s) vs. true Φ(s) on a grid of states
5. **Policy quality**: render/evaluate final policy qualitatively
6. **MSE**: compare MSE(V,return) of the two scenarios

## TODO

- Check if `self.phi_fn(obs) + f_theta` is correct or should we use prev obs
