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
                 phi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 action_cost_alpha: float = 0.0):
        self.cfg = cfg
        self.device = device
        self.writer = writer

        self.actor = Actor(obs_dim, act_dim, cfg.hidden_dim).to(device)
        self.critic = make_critic(cfg.condition, obs_dim, cfg.hidden_dim,
                                  phi_fn=phi_fn,
                                  zero_init_residual=cfg.zero_init_residual,
                                  feature_dim=cfg.feature_dim,
                                  action_cost_alpha=action_cost_alpha,
                                  gamma=cfg.gamma,
                                  n_mc_samples=cfg.n_mc_samples).to(device)

        # Wire actor's action-cost function into kernel critics when alpha > 0.
        if cfg.condition in ("C", "D") and action_cost_alpha > 0:
            self.critic.act_cost_fn = self.actor.action_cost

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

        is_c = cfg.condition == "C"
        is_d = cfg.condition == "D"

        # Conditions C and D both use geometrically-sampled (anchor, future) pairs.
        if is_c or is_d:
            anchor_obs, future_obs = self.buffer.get_contrastive_pairs(cfg.gamma)

        # Condition C only: refresh the kernel context buffer from the current rollout.
        if is_c:
            ctx_idx = torch.randperm(cfg.T, device=self.device)[:cfg.nce_context_size]
            self.critic.precompute_context(obs[ctx_idx])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        kernel_loss_val = torch.tensor(0.0, device=self.device)
        for _ in range(cfg.n_epochs):
            indices = torch.randperm(cfg.T, device=self.device)
            pair_perm = (torch.randperm(len(anchor_obs), device=self.device)
                         if (is_c or is_d) else None)

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

                # Kernel-specific auxiliary loss on a minibatch of (anchor, future) pairs
                if (is_c or is_d) and len(anchor_obs) > 0:
                    p_start = (i * cfg.batch_size) % len(anchor_obs)
                    p_idx = pair_perm[p_start: p_start + cfg.batch_size]
                    if len(p_idx) > 0:
                        a_obs = anchor_obs[p_idx]
                        f_obs = future_obs[p_idx]
                        if is_c:
                            kernel_loss_val = self.critic.nce_loss(a_obs, f_obs)
                            loss = loss + cfg.lambda_nce * kernel_loss_val
                        else:
                            kernel_loss_val = self.critic.nll_loss(a_obs, f_obs)
                            loss = loss #+ cfg.lambda_nll * kernel_loss_val

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

        self._log(policy_loss, value_loss, entropy_loss, kernel_loss_val)

    def _log(self, policy_loss, value_loss, entropy_loss, kernel_loss=None):
        s = self.global_step
        self.writer.add_scalar("loss/policy", policy_loss.item(), s)
        self.writer.add_scalar("loss/value", value_loss.item(), s)
        self.writer.add_scalar("loss/entropy", entropy_loss.item(), s)
        if self.cfg.condition == "B" and hasattr(self.critic, "residual_norm"):
            self.writer.add_scalar("critic/residual_norm", self.critic.residual_norm(), s)
        if self.cfg.condition == "C":
            if kernel_loss is not None:
                self.writer.add_scalar("loss/nce", kernel_loss.item(), s)
            self.writer.add_scalar("critic/I_mean", self.critic.I_mean(), s)
        if self.cfg.condition == "D" and kernel_loss is not None:
            self.writer.add_scalar("loss/nll", kernel_loss.item(), s)
