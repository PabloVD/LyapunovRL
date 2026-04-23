import numpy as np
import torch


class RolloutBuffer:
    """
    Stores one rollout of T steps and computes GAE advantages + discounted returns.
    """

    def __init__(self, T: int, obs_dim: int, act_dim: int, device: torch.device):
        self.T = T
        self.device = device

        self.obs = np.zeros((T, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, act_dim), dtype=np.float32)
        self.rewards = np.zeros(T, dtype=np.float32)
        self.dones = np.zeros(T, dtype=np.float32)
        self.log_probs = np.zeros(T, dtype=np.float32)
        self.values = np.zeros(T, dtype=np.float32)

        self.advantages = np.zeros(T, dtype=np.float32)
        self.returns = np.zeros(T, dtype=np.float32)

        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        gae = 0.0
        for t in reversed(range(self.T)):
            next_val = last_value if t == self.T - 1 else self.values[t + 1]
            next_non_term = 1.0 - (self.dones[t] if t < self.T - 1 else 0.0)
            delta = self.rewards[t] + gamma * next_val * next_non_term - self.values[t]
            gae = delta + gamma * lam * next_non_term * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_tensors(self):
        obs = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.float32, device=self.device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device).unsqueeze(-1)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        return obs, actions, log_probs, advantages, returns
