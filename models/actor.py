import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    """
    Gaussian policy for continuous action spaces.
    Outputs a squashed Normal distribution over actions.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy

    def action_cost(self, obs: torch.Tensor) -> torch.Tensor:
        """g(s) = E[‖a‖²] = ‖μ(s)‖² + Tr[Σ(s)] under the current Gaussian policy."""
        dist = self.forward(obs)
        return (dist.mean.pow(2) + dist.variance).sum(-1, keepdim=True)
