"""
Manager-only trainer for hierarchical MARL (single ego agent).

This is a self-contained implementation (no dependency on hmarl_zsc_actor_critic
or hmarl_manager_rmapppo) that matches the report’s architecture:
    - Trainable high-level manager selects a discrete skill every K steps.
    - Low-level controller (HMARLModel) is fixed/frozen and executes primitives
      conditioned on the chosen skill.
    - Only ego agent 0 is trained; partners can be frozen.

Intended to be plugged into a runner like overcooked_runner_hmarl_mng.py by
replacing the ego trainer with ManagerTrainerSingle and using its
get_actions_algorithm / update_buffer / training_step methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from loguru import logger

from zsceval.algorithms.hierarchical_marl_zsc.hmarl_policy import HMARLModel
from zsceval.algorithms.hierarchical_marl_zsc.utils.networks import ShareObsEncoder, ObsEncoder


def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


class _TrajEncoder(nn.Module):
    """Encodes a window of agent observations using the obs encoder + GRU."""

    def __init__(self, obs_shape: Tuple[int, ...], hidden_dim: int = 128):
        super().__init__()
        h, w, c = obs_shape
        self.enc = ObsEncoder(in_channels=c, obs_embedding_dim=hidden_dim, H=h, W=w)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """
        traj: (B, T, H, W, C_obs)
        returns: (B, hidden_dim)
        """
        B, T, H, W, C = traj.shape
        flat = traj.reshape(B * T, H, W, C)
        enc = self.enc(flat)  # (B*T, hidden_dim)
        enc = enc.reshape(B, T, -1)
        _, h = self.gru(enc)
        return h[-1]  # (B, hidden_dim)


class _ManagerNet(nn.Module):
    """
    Manager actor-critic with optional trajectory/context features.
    Inputs: shared obs, prev_skill one-hot, trajectory encoding, skill estimate.
    """

    def __init__(
        self,
        share_obs_shape: Tuple[int, ...],
        num_skills: int,
        traj_feat_dim: int,
        skill_est_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        h, w, c = share_obs_shape
        self.num_skills = num_skills
        self.enc = ShareObsEncoder(in_channels=c, state_embedding_dim=128, H=h, W=w)
        input_dim = 128 + num_skills + traj_feat_dim + skill_est_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, num_skills)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        share_obs: torch.Tensor,
        prev_skill: Optional[torch.Tensor],
        traj_feat: Optional[torch.Tensor],
        skill_est: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if share_obs.dim() == 5:
            share_obs = share_obs[:, 0] 
        enc = self.enc(share_obs)
        if prev_skill is None:
            prev_1h = torch.zeros(enc.shape[0], self.num_skills, device=enc.device)
        else:
            prev_skill = prev_skill.long().view(-1)
            prev_1h = F.one_hot(prev_skill, num_classes=self.num_skills).float()
        if traj_feat is None:
            traj_feat = torch.zeros(enc.shape[0], 0, device=enc.device)
        if skill_est is None:
            skill_est = torch.zeros(enc.shape[0], 0, device=enc.device)
        x = torch.cat([enc, prev_1h, traj_feat, skill_est], dim=-1)
        x = self.fc(x)
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value


class ManagerTrainerSingle:
    """
    Single-ego manager trainer with PPO-style update over skill periods.
    API mirrors HMARLTrainer for runner compatibility:
        - get_actions_algorithm
        - update_buffer
        - training_step
        - prep_rollout / prep_training
    """

    def __init__(self, config: Dict[str, Any], device: torch.device = torch.device("cpu")):
        cfg_tr = config["trainer"]
        cfg_m = config["model"]

        self.device = device
        self.num_agents = cfg_m.get("num_agents", 1)
        self.num_skills = cfg_m["num_skills"]
        self.num_actions = cfg_m["num_actions"]
        self.steps_per_assign = cfg_m["steps_per_assign"]
        self.gamma = cfg_tr.get("gamma", cfg_m.get("gamma", 0.99))
        self.gae_lambda = cfg_tr.get("gae_lambda", 0.95)
        self.clip_param = cfg_tr.get("clip_param", 0.2)
        self.entropy_coef = cfg_tr.get("entropy_coef", 0.0)
        self.value_loss_coef = cfg_tr.get("value_loss_coef", 0.5)
        self.max_grad_norm = cfg_tr.get("max_grad_norm", 0.5)
        self.ppo_epochs = cfg_tr.get("ppo_epochs", 4)
        self.num_mini_batch = cfg_tr.get("num_mini_batch", 1)
        self.lr = cfg_tr.get("manager_lr", 3e-4)

        self.hsd = HMARLModel(cfg_m, device=device)

        share_shape = (
            cfg_m["obs_height"],
            cfg_m["obs_width"],
            cfg_m["share_obs_channels"],
        )
        obs_shape = (
            cfg_m["obs_height"],
            cfg_m["obs_width"],
            cfg_m["obs_channels"],
        )
        self.traj_encoder = _TrajEncoder(obs_shape, hidden_dim=cfg_tr.get("traj_hidden_dim", 128))
        traj_feat_dim = self.traj_encoder.hidden_dim
        skill_est_dim = self.num_skills * 1 + 1
        self.manager = _ManagerNet(
            share_shape,
            self.num_skills,
            traj_feat_dim=traj_feat_dim,
            skill_est_dim=skill_est_dim,
            hidden_dim=cfg_tr.get("manager_hidden_dim", 256),
        )
        self.traj_encoder.to(device)
        self.manager.to(device)
        self.optim = Adam(self.manager.parameters(), lr=self.lr)

        self.current_skill: Optional[np.ndarray] = None
        self.prev_skill: Optional[np.ndarray] = None
        self.period_reward: Optional[np.ndarray] = None
        self._last_logp: Optional[torch.Tensor] = None
        self._last_value: Optional[torch.Tensor] = None
        self._last_traj_feat: Optional[torch.Tensor] = None
        self._last_skill_est: Optional[torch.Tensor] = None
        self._traj: List[Dict[str, torch.Tensor]] = []
        self._traj_buffer: List[np.ndarray] = []  # store recent shared obs per step

    @torch.no_grad()
    def prep_rollout(self):
        self.hsd.prep_rollout()
        self.manager.eval()

    @torch.no_grad()
    def prep_training(self):
        self.hsd.prep_training()
        self.manager.train()

    @torch.no_grad()
    def get_actions_algorithm(self, steps, obs, share_obs, available_actions):
        """
        Called each env step for ego agent.
        Inputs:
            obs:            (B, H, W, C)
            share_obs:      (B, H, W, C_share)
            available_actions: (B, num_actions)
        Returns:
            actions shaped for env: (B, 1, 1)
        """
        batch = obs.shape[0]
        if steps == 0 or self.period_reward is None:
            self.period_reward = np.zeros((batch,), dtype=np.float32)
            self._traj_buffer = []

        new_period = (steps % self.steps_per_assign == 0) or (self.current_skill is None)

        self._traj_buffer.append(obs[:, 0].copy())
        if len(self._traj_buffer) > self.steps_per_assign:
            self._traj_buffer.pop(0)

        if new_period:
            share_t = _to_tensor(share_obs, self.device)
            prev_skill_t = None if self.prev_skill is None else _to_tensor(self.prev_skill, self.device, dtype=torch.long)
            traj_feat = self._encode_traj()
            skill_est = self._estimate_partner_skill()
            logits, value = self.manager(share_t, prev_skill_t, traj_feat, skill_est)
            dist = Categorical(logits=logits)
            skill = dist.sample()
            logp = dist.log_prob(skill)

            self.current_skill = skill.cpu().numpy()
            self._last_logp = logp.detach()
            self._last_value = value.detach()
            self._last_traj_feat = traj_feat.detach()
            self._last_skill_est = skill_est.detach()
            self.period_reward = np.zeros((batch,), dtype=np.float32)

        # Align with HMARLModel.get_actions_low expectations: obs (B, N, H, W, C), skills (B, N)
        B, N = obs.shape[0], obs.shape[1]
        skills = np.repeat(self.current_skill[:, None], N, axis=1)  # (B, N)
        obs_exp = obs  # already (B, N, H, W, C)
        avail_exp = available_actions  # (B, N, A)
        actions = self.hsd.get_actions_low(obs_exp, avail_exp, skills)  # (B, N)
        return self._format_actions(actions)

    @staticmethod
    def _format_actions(actions: np.ndarray) -> np.ndarray:
        return np.expand_dims(actions, axis=-1)

    @torch.no_grad()
    def update_buffer(self, steps, obs, share_obs, actions, rewards, next_obs, next_share_obs, dones):
        """
        Accumulate rewards over skill period; store one transition per period.
        rewards: (B, num_agents, 1) or (B,) for ego; we use ego reward only.
        """
        r = np.asarray(rewards).squeeze()
        d = np.asarray(dones).squeeze()
        # logger.info(f"[hmarl_trainer_mng][step {steps}] raw rewards: {r}")
        if r.ndim > 1:
            r = r[:, 0]
        if d.ndim > 1:
            d = d[:, 0]
        self.period_reward += r

        end_period = ((steps + 1) % self.steps_per_assign == 0) or np.any(d)
        if end_period:
            share_t = _to_tensor(share_obs, self.device)
            next_share_t = _to_tensor(next_share_obs, self.device)
            skill_t = _to_tensor(self.current_skill, self.device, dtype=torch.long)
            rew_t = _to_tensor(self.period_reward, self.device)
            done_t = _to_tensor(d.astype(np.float32), self.device)
            traj_feat_t = self._last_traj_feat if self._last_traj_feat is not None else torch.zeros(
                (share_t.shape[0], self.traj_encoder.hidden_dim), device=self.device
            )
            skill_est_t = self._last_skill_est if self._last_skill_est is not None else torch.zeros(
                (share_t.shape[0], self.num_skills), device=self.device
            )

            self._traj.append(
                {
                    "share_obs": share_t,
                    "next_share_obs": next_share_t,
                    "skill": skill_t,
                    "reward": rew_t,
                    "done": done_t,
                    "log_prob": self._last_logp,
                    "value": self._last_value,
                    "traj_feat": traj_feat_t,
                    "skill_est": skill_est_t,
                }
            )
            self.prev_skill = self.current_skill
            self.period_reward = np.zeros_like(self.period_reward)

    def training_step(self, _unused: int = 0) -> Dict[str, float]:
        if not self._traj:
            return {}

        share_obs = torch.cat([t["share_obs"] for t in self._traj], dim=0)
        next_share_obs = torch.cat([t["next_share_obs"] for t in self._traj], dim=0)
        skills = torch.cat([t["skill"] for t in self._traj], dim=0)
        rewards = torch.cat([t["reward"] for t in self._traj], dim=0)
        dones = torch.cat([t["done"] for t in self._traj], dim=0)
        old_logp = torch.cat([t["log_prob"] for t in self._traj], dim=0)
        old_values = torch.cat([t["value"] for t in self._traj], dim=0).squeeze(-1)
        traj_feats = torch.cat([t["traj_feat"] for t in self._traj], dim=0)
        skill_ests = torch.cat([t["skill_est"] for t in self._traj], dim=0)

        with torch.no_grad():
            _, next_values = self.manager(next_share_obs, prev_skill=skills, traj_feat=traj_feats, skill_est=skill_ests)
            next_values = next_values.squeeze(-1)

        adv, ret = self._compute_gae(
            rewards, dones, old_values, next_values, gamma=self.gamma, lam=self.gae_lambda
        )

        num_samples = ret.shape[0]
        inds = np.arange(num_samples)
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(inds)
            mb_size = max(1, num_samples // self.num_mini_batch)
            for start in range(0, num_samples, mb_size):
                end = start + mb_size
                mb_inds = inds[start:end]
                mb_obs = share_obs[mb_inds]
                mb_skill = skills[mb_inds]
                mb_old_logp = old_logp[mb_inds]
                mb_adv = adv[mb_inds]
                mb_ret = ret[mb_inds]
                mb_old_val = old_values[mb_inds]

                logits, value = self.manager(mb_obs, prev_skill=mb_skill, traj_feat=traj_feats[mb_inds], skill_est=skill_ests[mb_inds])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(mb_skill)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value = value.squeeze(-1)
                value_clipped = mb_old_val + (value - mb_old_val).clamp(-self.clip_param, self.clip_param)
                value_loss = 0.5 * torch.max((mb_ret - value) ** 2, (mb_ret - value_clipped) ** 2).mean()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.manager.parameters(), self.max_grad_norm)
                self.optim.step()

                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy.item()

        self._traj = []
        n_batches = max(1, self.ppo_epochs * self.num_mini_batch)
        return {
            "manager_policy_loss": policy_loss_epoch / n_batches,
            "manager_value_loss": value_loss_epoch / n_batches,
            "manager_entropy": entropy_epoch / n_batches,
        }

    @staticmethod
    def _compute_gae(rewards, dones, values, next_values, gamma, lam):
        advantages = torch.zeros_like(rewards, device=rewards.device)
        gae = 0.0
        for t in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values[t] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return advantages, returns

    def save(self, step: int, save_path: str) -> None:
        torch.save(self.manager.state_dict(), f"{save_path}/manager_{step}.pt")

    def load(self, path: str, map_location=None) -> None:
        state = torch.load(path, map_location=map_location or self.device)
        self.manager.load_state_dict(state)

    def _encode_traj(self) -> torch.Tensor:
        """
        Encode the last steps_per_assign ego observations into a trajectory feature.
        Pads with zeros if not enough history.
        """
        if not self._traj_buffer:
            batch = getattr(self, "period_reward", np.zeros((1,), dtype=np.float32)).shape[0]
            return torch.zeros(batch, self.traj_encoder.hidden_dim, device=self.device)
        traj_np = np.stack(self._traj_buffer, axis=0)  # (T, B, H, W, C_obs)
        traj_np = np.transpose(traj_np, (1, 0, 2, 3, 4))  # (B, T, H, W, C_obs)
        B, T, H, W, C = traj_np.shape
        if T < self.steps_per_assign:
            pad = np.zeros((B, self.steps_per_assign - T, H, W, C), dtype=traj_np.dtype)
            traj_np = np.concatenate([traj_np, pad], axis=1)
        traj_t = _to_tensor(traj_np, self.device)
        feat = self.traj_encoder(traj_t)
        return feat

    def _estimate_partner_skill(self) -> torch.Tensor:
        """
        Estimate partner skill distribution using the pretrained decoder.
        This treats partners as all agents except ego and pools their per-skill probs.
        """
        if not self._traj_buffer:
            batch = getattr(self, "period_reward", np.zeros((1,), dtype=np.float32)).shape[0]
            return torch.zeros(batch, self.num_skills, device=self.device)

        # _traj_buffer: list length T of ego obs (B, H, W, C_obs).
        # If partner observations are available in runner, extend this to (B, P, H, W, C_obs).
        traj_np = np.stack(self._traj_buffer, axis=0)  # (T, B, H, W, C_obs)
        traj_np = np.transpose(traj_np, (1, 0, 2, 3, 4))  # (B, T, H, W, C_obs)
        B, T, H, W, C = traj_np.shape

        # Treat partners as a flattened dim; here we only have ego, so P=1. If you add partners, reshape accordingly.
        P = 1
        traj_flat = traj_np.reshape(B * P, T, H, W, C)
        traj_t = torch.as_tensor(traj_flat, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            Bflat, Tflat, _, _, _ = traj_t.shape
            traj_frames = traj_t.reshape(Bflat * Tflat, H, W, C)
            obs_encoded = self.hsd.obs_encoder(traj_frames) 
            obs_seq = obs_encoded.reshape(Bflat, Tflat, -1)

            down = obs_seq[:, :: self.hsd.traj_skip, :]
            if self.hsd.obs_truncate_length:
                down = down[:, :, : self.hsd.obs_truncate_length]
            if self.hsd.use_state_difference:
                down = down[:, 1:, :] - down[:, :-1, :]

            _, probs = self.hsd.decoder(down)  # (Bflat, num_skills)

        # Flatten partners into feature: (B, P*num_skills)
        probs = probs.reshape(B, P * self.num_skills)
        count_feat = torch.full((B, 1), float(P), device=self.device) / max(1.0, float(self.num_agents - 1))
        pooled = torch.cat([probs, count_feat], dim=1)  # (B, P*num_skills + 1)
        return pooled
