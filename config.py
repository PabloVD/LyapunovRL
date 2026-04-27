from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    condition: str = "A"        # "A" = standard critic, "B" = Lyapunov-structured, "C" = bilinear kernel
    seed: int = 0
    total_steps: int = 300_000
    T: int = 2048               # rollout length
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    device: str = "cuda"
    env_id: str = "Pendulum-v1"
    native_reward: bool = False   # use env's original reward instead of Lyapunov
    zero_init_residual: bool = True  # zero-init residual head in CriticB so V≈Φ at t=0
    normalize_obs: bool = False   # wrap env with running-stats NormalizeObservation
    target_velocity: float = 1.0  # target forward speed for LyapunovAnt (m/s)
    lambda_nce: float = 1.0      # weight of InfoNCE contrastive loss (condition C only)
    feature_dim: int = 64        # kernel embedding dimension (condition C only)
    nce_context_size: int = 512  # number of context states used for kernel integration
    log_interval: int = 1       # log every N updates
    run_name: str = ""          # auto-set in train.py if empty

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            data = yaml.safe_load(f)
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False, sort_keys=False)
