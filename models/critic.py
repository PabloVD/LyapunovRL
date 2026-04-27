import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CriticC(nn.Module):
    """
    Condition C: V(s) = Φ(s) − I(s)

    I(s) = Σ_{s'∈B} softmax( φ_θ(s)·ψ_θ(s') / √d ) · Φ(s')

    The bilinear kernel φ_θ(s)·ψ_θ(s') approximates 𝒯(s'|s), the discounted
    future-state distribution. I(s) is its expectation under Φ.

    Training uses two losses:
      1. InfoNCE on (anchor, future) pairs sampled with geometric lag k~Geom(1−γ)
      2. Standard MSE value loss on GAE returns (gradients flow only through query_net)

    A context buffer of representative states is refreshed after each rollout.
    """

    def __init__(
        self,
        obs_dim: int,
        phi_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_dim: int = 64,
        feature_dim: int = 64,
    ):
        super().__init__()
        self.phi_fn = phi_fn
        self.feature_dim = feature_dim

        self.query_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.key_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
        )
        # Zero-init query output layer → I(s) ≈ 0 at start → V ≈ Φ
        nn.init.zeros_(self.query_net[-1].weight)
        nn.init.zeros_(self.query_net[-1].bias)

        # Context buffer (filled by precompute_context before each rollout)
        self.register_buffer("ctx_key_embs", torch.zeros(1, feature_dim))
        self.register_buffer("ctx_phi_vals", torch.zeros(1))

    @torch.no_grad()
    def precompute_context(self, ctx_obs: torch.Tensor) -> None:
        """Cache key embeddings and Φ values for a set of context states."""
        self.ctx_key_embs = self.key_net(ctx_obs)       # (N, d)
        self.ctx_phi_vals = self.phi_fn(ctx_obs).squeeze(-1)  # (N,)

    def _compute_I(self, obs: torch.Tensor, key_embs: torch.Tensor,
                   phi_vals: torch.Tensor) -> torch.Tensor:
        """Soft-attention integration: I(s) = softmax(φ(s)·ψ(B)/√d) · Φ(B)."""
        q = self.query_net(obs)                                # (B, d)
        logits = q @ key_embs.T / math.sqrt(self.feature_dim) # (B, N)
        weights = F.softmax(logits, dim=-1)                    # (B, N)
        return (weights @ phi_vals.unsqueeze(-1)).squeeze(-1)  # (B,)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Use cached context (key embeddings are detached for value backprop)
        I_s = self._compute_I(obs, self.ctx_key_embs.detach(), self.ctx_phi_vals.detach())
        return self.phi_fn(obs) - I_s.unsqueeze(-1)

    def nce_loss(self, anchor_obs: torch.Tensor, future_obs: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE loss over a batch of (anchor, future) pairs.
        Both query_net and key_net receive gradients here.
        Treats each pair as its own positive class; all others are negatives.
        """
        q = self.query_net(anchor_obs)   # (N, d)
        k = self.key_net(future_obs)     # (N, d)
        logits = q @ k.T / math.sqrt(self.feature_dim)  # (N, N)
        labels = torch.arange(len(anchor_obs), device=anchor_obs.device)
        return F.cross_entropy(logits, labels)

    def I_mean(self) -> float:
        return self.ctx_phi_vals.mean().item()


def make_critic(
    condition: str,
    obs_dim: int,
    hidden_dim: int = 64,
    phi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    zero_init_residual: bool = False,
    feature_dim: int = 64,
) -> nn.Module:
    if condition == "A":
        return CriticA(obs_dim, hidden_dim)
    elif condition == "B":
        if phi_fn is None:
            raise ValueError("Condition B requires phi_fn — pass env_class.phi_torch.")
        return CriticB(obs_dim, phi_fn, hidden_dim, zero_init_residual=zero_init_residual)
    elif condition == "C":
        if phi_fn is None:
            raise ValueError("Condition C requires phi_fn — pass env_class.phi_torch.")
        return CriticC(obs_dim, phi_fn, hidden_dim, feature_dim=feature_dim)
    raise ValueError(f"Unknown condition: {condition!r}. Use 'A', 'B', or 'C'.")
