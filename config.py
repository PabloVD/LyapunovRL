from dataclasses import dataclass


@dataclass
class Config:
    condition: str = "A"        # "A" = standard critic, "B" = Lyapunov-structured
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
    log_interval: int = 1       # log every N updates
    run_name: str = ""          # auto-set in train.py if empty
