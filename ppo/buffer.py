import numpy as np
import torch


class RolloutBuffer:
    """
    Stores one rollout of T steps and computes GAE advantages + discounted returns.

    Correctly handles the terminated vs truncated distinction:
      - terminated: true episode end (absorbing state) — next value is 0.
      - truncated:  timeout — episode is cut off but the state still has value.
        The caller must supply V(final_obs) as next_value before resetting the env.
    """

    def __init__(self, T: int, obs_dim: int, act_dim: int, device: torch.device):
        self.T = T
        self.device = device

        self.obs = np.zeros((T, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, act_dim), dtype=np.float32)
        self.rewards = np.zeros(T, dtype=np.float32)
        self.episode_ends = np.zeros(T, dtype=np.float32)   # 1 at any episode boundary
        self.next_values = np.zeros(T, dtype=np.float32)    # correct V(s_{t+1}) per step
        self.log_probs = np.zeros(T, dtype=np.float32)
        self.values = np.zeros(T, dtype=np.float32)

        self.advantages = np.zeros(T, dtype=np.float32)
        self.returns = np.zeros(T, dtype=np.float32)

        self.ptr = 0

    def add(self, obs, action, reward, terminated: bool, truncated: bool,
            log_prob: float, value: float, next_value: float):
        """
        next_value: V(s_{t+1}) pre-computed by the caller.
          - non-done step:   value of the next obs in the rollout (values[ptr+1])
            → pass 0 here; compute_gae will fill it from self.values[t+1].
          - truncated step:  V(final_obs) computed before env.reset().
          - terminated step: 0 (absorbing state, no future reward).
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.episode_ends[self.ptr] = float(terminated or truncated)
        self.next_values[self.ptr] = next_value
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        # Fill next_values for non-episode-end steps from the buffer itself.
        for t in range(self.T - 1):
            if not self.episode_ends[t]:
                self.next_values[t] = self.values[t + 1]
        if not self.episode_ends[self.T - 1]:
            self.next_values[self.T - 1] = last_value  # end of rollout, not episode end

        gae = 0.0
        for t in reversed(range(self.T)):
            delta = self.rewards[t] + gamma * self.next_values[t] - self.values[t]
            # Reset GAE accumulation at any episode boundary.
            gae = delta + gamma * lam * (1.0 - self.episode_ends[t]) * gae
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

    def get_contrastive_pairs(self, gamma: float):
        """
        Build (anchor, future) pairs for InfoNCE training of the kernel 𝒯(s'|s).

        For each step t, the positive future state is s_{t+k} where k ~ Geom(1−γ)
        capped at the next episode boundary. Steps at episode ends are discarded.

        Returns: (anchor_obs, future_obs) float32 tensors on self.device.
        """
        anchors, futures = [], []
        ends = np.where(self.episode_ends > 0)[0]  # sorted episode-boundary indices

        for t in range(self.T - 1):
            # Steps ahead until (and including) the next episode boundary
            next_end_idx = int(np.searchsorted(ends, t, side="right"))
            if next_end_idx < len(ends):
                max_k = int(ends[next_end_idx]) - t
            else:
                max_k = self.T - 1 - t
            if max_k < 1:
                continue  # t is an episode end; no valid same-trajectory future

            # k ~ Geom(1−γ), k ≥ 1, capped so we stay in the same episode
            k = min(int(np.random.geometric(1.0 - gamma)), max_k)
            anchors.append(self.obs[t])
            futures.append(self.obs[t + k])

        if not anchors:
            # Fallback: consecutive pairs (degenerate but avoids empty tensors)
            anchors = list(self.obs[:-1])
            futures = list(self.obs[1:])

        anchor_t = torch.tensor(np.array(anchors), dtype=torch.float32, device=self.device)
        future_t = torch.tensor(np.array(futures), dtype=torch.float32, device=self.device)
        return anchor_t, future_t
