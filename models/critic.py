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
    Condition C: V(s) = Φ(s) − I_Φ(s) − α·g(s) − α·γ/(1−γ)·I_g(s)

    where:
      I_Φ(s) = E_{s'~𝒯}[Φ(s')]           (Lyapunov integral)
      I_g(s) = E_{s'~𝒯}[g(s')]             (action-cost integral)
      g(s)   = ‖μ(s)‖² + Tr[Σ(s)]          (expected squared action)
      𝒯(s'|s) ∝ exp(φ_θ(s)·ψ_θ(s')/√d)   (learned bilinear kernel)

    Both integrals reuse the same softmax kernel weights.
    The action-cost term is optional (action_cost_alpha=0 disables it).

    Training uses two losses:
      1. InfoNCE on (anchor, future) pairs sampled with geometric lag k~Geom(1−γ)
      2. Standard MSE value loss on GAE returns (gradients through query_net only)

    A context buffer of states is refreshed from the rollout after each update.
    act_cost_fn must be set externally (e.g. to actor.action_cost) when alpha > 0.
    """

    def __init__(
        self,
        obs_dim: int,
        phi_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_dim: int = 64,
        feature_dim: int = 64,
        action_cost_alpha: float = 0.0,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.phi_fn = phi_fn
        self.feature_dim = feature_dim
        self.action_cost_alpha = action_cost_alpha
        self.gamma = gamma

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

        # Context cache — plain attributes (not registered buffers), excluded from state_dict.
        self.ctx_key_embs: torch.Tensor | None = None
        self.ctx_phi_vals: torch.Tensor | None = None
        self.ctx_act_costs: torch.Tensor | None = None

        # Set externally by the trainer to actor.action_cost when alpha > 0.
        self.act_cost_fn: Callable[[torch.Tensor], torch.Tensor] | None = None

    @torch.no_grad()
    def precompute_context(self, ctx_obs: torch.Tensor) -> None:
        """Cache key embeddings, Φ values, and (if configured) action costs for context states."""
        self.ctx_key_embs = self.key_net(ctx_obs)              # (N, d)
        self.ctx_phi_vals = self.phi_fn(ctx_obs).squeeze(-1)   # (N,)
        if self.action_cost_alpha > 0 and self.act_cost_fn is not None:
            self.ctx_act_costs = self.act_cost_fn(ctx_obs).squeeze(-1)  # (N,)
        else:
            self.ctx_act_costs = None

    def _kernel_weights(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute softmax kernel weights φ(s)·ψ(B)/√d for the cached context."""
        q = self.query_net(obs)                                          # (B, d)
        logits = q @ self.ctx_key_embs.detach().T / math.sqrt(self.feature_dim)  # (B, N)
        return F.softmax(logits, dim=-1)                                 # (B, N)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.ctx_key_embs is None:
            # No context yet — fall back to V ≈ Φ (matches zero-init intent)
            return self.phi_fn(obs)

        weights = self._kernel_weights(obs)   # (B, N)

        # Lyapunov integral: I_Φ(s) = E_{s'~𝒯}[Φ(s')]
        I_phi = (weights @ self.ctx_phi_vals.detach().unsqueeze(-1)).squeeze(-1)  # (B,)
        V = self.phi_fn(obs) - I_phi.unsqueeze(-1)

        # Action-cost term: −α·g(s) − α·γ/(1−γ)·I_g(s)
        if (self.action_cost_alpha > 0
                and self.act_cost_fn is not None
                and self.ctx_act_costs is not None):
            alpha = self.action_cost_alpha
            coeff = self.gamma / (1.0 - self.gamma)
            I_g = (weights @ self.ctx_act_costs.detach().unsqueeze(-1)).squeeze(-1)  # (B,)
            with torch.no_grad():
                g_s = self.act_cost_fn(obs).squeeze(-1)  # (B,) — detached: actor gradient via policy loss only
            V = V - alpha * (g_s + coeff * I_g).unsqueeze(-1)

        return V

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
        return self.ctx_phi_vals.mean().item() if self.ctx_phi_vals is not None else 0.0


def make_critic(
    condition: str,
    obs_dim: int,
    hidden_dim: int = 64,
    phi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    zero_init_residual: bool = False,
    feature_dim: int = 64,
    action_cost_alpha: float = 0.0,
    gamma: float = 0.99,
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
        return CriticC(obs_dim, phi_fn, hidden_dim, feature_dim=feature_dim,
                       action_cost_alpha=action_cost_alpha, gamma=gamma)
    raise ValueError(f"Unknown condition: {condition!r}. Use 'A', 'B', or 'C'.")
