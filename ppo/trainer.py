from typing import Callable

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.actor import Actor
from models.critic import make_critic
from ppo.buffer import RolloutBuffer


class PPOTrainer:
    def __init__(self, cfg, device: torch.device, writer: SummaryWriter,
                 obs_dim: int, act_dim: int,
                 phi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.cfg = cfg
        self.device = device
        self.writer = writer

        self.actor = Actor(obs_dim, act_dim, cfg.hidden_dim).to(device)
        self.critic = make_critic(cfg.condition, obs_dim, cfg.hidden_dim,
                                  phi_fn=phi_fn,
                                  zero_init_residual=cfg.zero_init_residual,
                                  feature_dim=cfg.feature_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.buffer = RolloutBuffer(cfg.T, obs_dim, act_dim, device)
        self.global_step = 0

    @torch.no_grad()
    def get_action_and_value(self, obs_np):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, _ = self.actor.get_action(obs)
        value = self.critic(obs)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze().item(),
            value.squeeze().item(),
        )

    def update(self, last_value: float):
        cfg = self.cfg
        self.buffer.compute_gae(last_value, cfg.gamma, cfg.lam)
        obs, actions, old_log_probs, advantages, returns = self.buffer.get_tensors()

        # Condition C: refresh context buffer from current rollout and build contrastive pairs
        is_c = cfg.condition == "C"
        if is_c:
            ctx_idx = torch.randperm(cfg.T, device=self.device)[:cfg.nce_context_size]
            self.critic.precompute_context(obs[ctx_idx])
            anchor_obs, future_obs = self.buffer.get_contrastive_pairs(cfg.gamma)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        nce_loss_val = torch.tensor(0.0, device=self.device)
        for _ in range(cfg.n_epochs):
            indices = torch.randperm(cfg.T, device=self.device)

            # Shuffle contrastive pairs once per epoch (independent permutation)
            if is_c:
                nce_perm = torch.randperm(len(anchor_obs), device=self.device)

            for i, start in enumerate(range(0, cfg.T, cfg.batch_size)):
                idx = indices[start: start + cfg.batch_size]

                b_obs = obs[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_advantages = advantages[idx]
                b_returns = returns[idx]

                new_log_probs, entropy = self.actor.evaluate_actions(b_obs, b_actions)
                ratio = (new_log_probs - b_old_log_probs).exp()

                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_pred = self.critic(b_obs)
                value_loss = nn.functional.mse_loss(value_pred, b_returns)

                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.entropy_coef * entropy_loss

                if is_c:
                    # Sample a minibatch of contrastive pairs for this step
                    nce_start = (i * cfg.batch_size) % max(len(anchor_obs), 1)
                    nce_idx = nce_perm[nce_start: nce_start + cfg.batch_size]
                    if len(nce_idx) > 0:
                        nce_loss_val = self.critic.nce_loss(
                            anchor_obs[nce_idx], future_obs[nce_idx]
                        )
                        loss = loss + cfg.lambda_nce * nce_loss_val

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

        self._log(policy_loss, value_loss, entropy_loss, nce_loss_val)

    def _log(self, policy_loss, value_loss, entropy_loss, nce_loss=None):
        s = self.global_step
        self.writer.add_scalar("loss/policy", policy_loss.item(), s)
        self.writer.add_scalar("loss/value", value_loss.item(), s)
        self.writer.add_scalar("loss/entropy", entropy_loss.item(), s)
        if self.cfg.condition == "B" and hasattr(self.critic, "residual_norm"):
            self.writer.add_scalar("critic/residual_norm", self.critic.residual_norm(), s)
        if self.cfg.condition == "C":
            if nce_loss is not None:
                self.writer.add_scalar("loss/nce", nce_loss.item(), s)
            self.writer.add_scalar("critic/I_mean", self.critic.I_mean(), s)
