from typing import Callable

import torch
import torch.nn as nn


class CriticA(nn.Module):
    """Condition A: standard value network. Learns V(s) entirely from data."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticB(nn.Module):
    """
    Condition B: structured value network.
    V(s) = Φ(s) + f_θ(s), where only f_θ is learned.

    phi_fn is supplied by the environment class so this critic is not
    coupled to any specific Lyapunov function.
    The last layer is zero-initialized so V ≈ Φ at training start.
    """

    def __init__(self, obs_dim: int, phi_fn: Callable[[torch.Tensor], torch.Tensor],
                 hidden_dim: int = 64, zero_init_residual: bool = False):
        super().__init__()
        self.phi_fn = phi_fn
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        if zero_init_residual:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        f_theta = self.residual_head(self.net(obs))
        return self.phi_fn(obs) + f_theta

    def residual_norm(self) -> float:
        return self.residual_head.weight.data.norm().item()


def make_critic(
    condition: str,
    obs_dim: int,
    hidden_dim: int = 64,
    phi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    zero_init_residual: bool = False,
) -> nn.Module:
    if condition == "A":
        return CriticA(obs_dim, hidden_dim)
    elif condition == "B":
        if phi_fn is None:
            raise ValueError("Condition B requires phi_fn — pass env_class.phi_torch.")
        return CriticB(obs_dim, phi_fn, hidden_dim, zero_init_residual=zero_init_residual)
    raise ValueError(f"Unknown condition: {condition!r}. Use 'A' or 'B'.")
