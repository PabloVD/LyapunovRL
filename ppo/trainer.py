import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.actor import Actor
from models.critic import make_critic
from ppo.buffer import RolloutBuffer


class PPOTrainer:
    def __init__(self, cfg, device: torch.device, writer: SummaryWriter):
        self.cfg = cfg
        self.device = device
        self.writer = writer

        obs_dim = 3
        act_dim = 1

        self.actor = Actor(obs_dim, act_dim, cfg.hidden_dim).to(device)
        self.critic = make_critic(cfg.condition, obs_dim, cfg.hidden_dim).to(device)

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

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(cfg.n_epochs):
            indices = torch.randperm(cfg.T, device=self.device)
            for start in range(0, cfg.T, cfg.batch_size):
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

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

        self._log(policy_loss, value_loss, entropy_loss)

    def _log(self, policy_loss, value_loss, entropy_loss):
        s = self.global_step
        self.writer.add_scalar("loss/policy", policy_loss.item(), s)
        self.writer.add_scalar("loss/value", value_loss.item(), s)
        self.writer.add_scalar("loss/entropy", entropy_loss.item(), s)
        if self.cfg.condition == "B" and hasattr(self.critic, "residual_norm"):
            self.writer.add_scalar("critic/residual_norm", self.critic.residual_norm(), s)
