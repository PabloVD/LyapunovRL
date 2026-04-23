import torch
import torch.nn as nn
from lyapunov.functions import phi_torch


class CriticA(nn.Module):
    """
    Condition A: standard value network. Learns V(s) entirely from data.
    """

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

    The last layer is zero-initialized so that V ≈ Φ at the start of training,
    giving the network a physically meaningful starting point.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.net(obs)
        f_theta = self.residual_head(features)
        phi = phi_torch(obs)
        return phi + f_theta

    def residual_norm(self) -> float:
        return self.residual_head.weight.data.norm().item()


def make_critic(condition: str, obs_dim: int, hidden_dim: int = 64) -> nn.Module:
    if condition == "A":
        return CriticA(obs_dim, hidden_dim)
    elif condition == "B":
        return CriticB(obs_dim, hidden_dim)
    raise ValueError(f"Unknown condition: {condition!r}. Use 'A' or 'B'.")
