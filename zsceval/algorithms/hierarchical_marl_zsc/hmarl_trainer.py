from __future__ import annotations
from collections import deque
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from zsceval.runner.shared.overcooked_runner import OvercookedRunner
from zsceval.algorithms.hierarchical_marl_zsc.hmarl_policy import HMARLModel
from zsceval.algorithms.hierarchical_marl_zsc.utils.replay_buffer import Replay_Buffer


# Trainer Class Compatible with ZSC-Eval
# In training, it is wrapped with simplest runner which not compatible with base_runner
# After training, it provides functions other policies can use (decoder, assign_skills, get_actions, ...)
#                 it has function which creates fixed Agent Instances
class HMARLTrainer(OvercookedRunner):
    """Wrapper to bridge ZSC env messaging with HMARL policy/trainer."""

    def __init__(self, config, device=torch.device("cpu")):

        # Extract structured configs
        cfg_tr = config["trainer"]
        cfg_m = config["model"]

        # Seeding
        seed = cfg_tr["seed"]
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Load trainer params
        self.N_train = cfg_tr["N_train"]
        self.N_eval = cfg_tr["N_eval"]
        self.period = cfg_tr["period"]
        self.buffer_size = cfg_tr["buffer_size"]
        self.batch_size = cfg_tr["batch_size"]
        self.pretrain_episodes = cfg_tr["pretrain_episodes"]
        self.steps_per_train = cfg_tr["steps_per_train"]

        # Exploration params
        self.epsilon_start = cfg_tr["epsilon_start"]
        self.epsilon_end = cfg_tr["epsilon_end"]
        self.epsilon_div = cfg_tr["epsilon_div"]
        self.epsilon = self.epsilon_start
        self.epsilon_step = (self.epsilon_start - self.epsilon_end) / float(
            self.epsilon_div
        )

        # Reward mixing parameters
        self.alpha = cfg_tr["alpha_start"]
        self.alpha_end = cfg_tr["alpha_end"]
        self.alpha_step = cfg_tr["alpha_step"]
        self.alpha_threshold = cfg_tr["alpha_threshold"]

        # Skills
        self.N_skills = cfg_tr["N_skills"]
        self.steps_per_assign = cfg_tr["steps_per_assign"]
        self.decoder_training_threshold = cfg_tr["decoder_training_threshold"]

        # Load model parameters
        state_dim = cfg_m["state_dim"]
        num_actions = cfg_m["num_actions"]
        obs_dim = cfg_m["obs_dim"]
        self.num_agents = cfg_m["num_agents"]
        self.num_actions = cfg_m["num_actions"]

        # ---- Build HMARL Policy ----
        self.hsd = HMARLModel(cfg_m, device=device)

        # ---- Replay buffers ----
        self.buf_high = Replay_Buffer(size=self.buffer_size)
        self.buf_low = Replay_Buffer(size=self.buffer_size)

        # ---- Internal variables ----
        self.current_skills = np.zeros((self.batch_size, self.num_agents), dtype=int)
        self.intrinsic_rewards = np.zeros((self.batch_size, self.num_agents))

        # Per-agent trajectory sliding window: deque with maxlen = steps_per_assign
        self.traj_per_agent = [
            [deque(maxlen=self.steps_per_assign) for _ in range(self.num_agents)]
            for _ in range(self.batch_size)
        ]

        self.dataset = []
        self.obs_h = None
        self.share_obs_h = None

        # High-level cumulative reward per env (scalar, not per-agent)
        self.rewards_high = np.zeros((self.batch_size,), dtype=float)

        # Track aggregated high-level rewards per skill for logging
        self.episode_high_level_rewards = []

        # Global env step tracker
        self.total_env_steps = 0

        # First training step happens after pretrain_episodes env steps
        self.next_train_step = self.pretrain_episodes

    ## --- Core functions that is run only in shared overcooked HMARL runner --- ##

    # Update Q_low, Q_high, decoder based on internal buffer and counter using internals
    def training_step(self, _unused: int = 0):
        # Set training mode for policy
        self.prep_training()

        updates_done = 0
        expected_prob = None

        # Perform as many updates as needed to catch up with env steps
        while self.total_env_steps >= self.next_train_step:
            if self.total_env_steps >= self.pretrain_episodes:
                # Low-level update
                batch_low = self.buf_low.sample_batch(self.batch_size)
                self.hsd.train_policy_low(batch_low)

                # High-level update
                batch_high = self.buf_high.sample_batch(self.batch_size)
                self.hsd.train_policy_high(batch_high)
                updates_done += 1

                # Decoder update if dataset is large enough
                if len(self.dataset) >= self.decoder_training_threshold:
                    expected_prob = self.hsd.train_decoder(self.dataset)
                    self.dataset = []

            self.next_train_step += self.steps_per_train

        # --- EPSILON / ALPHA SCHEDULE ---
        if updates_done > 0 and self.total_env_steps >= self.pretrain_episodes:
            self.epsilon = max(
                self.epsilon_end, self.epsilon - self.epsilon_step * updates_done
            )

        if self.alpha < self.alpha_threshold and updates_done > 0:
            self.alpha = min(
                self.alpha_threshold,
                self.alpha + self.alpha_step * updates_done,
            )

        # Snapshot the high-level rewards collected over skill periods within the episode
        if self.episode_high_level_rewards:
            high_level_reward_mean = float(np.mean(self.episode_high_level_rewards[-5:]))  # FIXME: mean over recent high level rewards
        else:
            high_level_reward_mean = 0.0

        # --- Logging ---
        train_infos = {
            "epsilon": float(self.epsilon),
            "alpha": float(self.alpha),
            "intrinsic_reward_mean": float(np.mean(self.intrinsic_rewards)),
            "high_level_reward_mean": high_level_reward_mean,
        }

        if expected_prob is not None:
            train_infos["decoder_expected_prob"] = float(expected_prob)

        flat_skills = self.current_skills.flatten()
        counts = np.bincount(flat_skills, minlength=self.N_skills)
        if counts.sum() > 0:
            train_infos["skill_usage"] = (counts / counts.sum()).tolist()
        else:
            train_infos["skill_usage"] = [0.0 for _ in range(self.N_skills)]

        return train_infos

    # Update buffer and accumulated high level rewards based on environment step
    @torch.no_grad()
    def update_buffer(
        self, steps, obs, share_obs, actions, rewards, next_obs, next_share_obs, dones
    ):
        # steps: step within episode

        # ---------------------------------------------------
        # 1) Batch size sanity check
        # ---------------------------------------------------
        obs_np = np.asarray(obs)
        incoming_batch = obs_np.shape[0]

        if incoming_batch != self.batch_size:
            raise ValueError(
                f"[update_buffer] Incoming batch {incoming_batch} != trainer.batch_size {self.batch_size}"
            )

        # ---------------------------------------------------
        # 2) Normalize rewards shape: expect (batch_size, num_agents)
        # ---------------------------------------------------
        rewards = np.asarray(rewards)
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)  # (batch, agents)

        if rewards.shape != (self.batch_size, self.num_agents):
            raise ValueError(
                f"[update_buffer] Expected rewards shape {(self.batch_size, self.num_agents)}, "
                f"got {rewards.shape}"
            )

        # ---------------------------------------------------
        # 3) Normalize dones: env-level done flag per rollout
        # ---------------------------------------------------
        dones = np.asarray(dones)
        # common format: (batch, num_agents, 1) with same value for all agents
        if dones.ndim == 3 and dones.shape[-1] == 1:
            dones = dones.squeeze(-1)  # (batch, agents)
        if dones.ndim == 2:
            # env done if any agent is done
            dones_env = np.any(dones > 0.5, axis=1).astype(np.float32)  # (batch,)
        elif dones.ndim == 1:
            dones_env = (dones > 0.5).astype(np.float32)
        else:
            raise ValueError(
                f"[update_buffer] Unexpected dones shape: {dones.shape}"
            )

        # ---------------------------------------------------
        # 4) Update per-agent sliding window trajectories (for decoder/IR)
        #    Push current obs; deque(maxlen) will drop oldest automatically.
        # ---------------------------------------------------
        for b in range(self.batch_size):
            for ag in range(self.num_agents):
                self.traj_per_agent[b][ag].append(obs[b][ag])

        # ---------------------------------------------------
        # 5) Compute intrinsic rewards if enough history
        # ---------------------------------------------------
        self.intrinsic_rewards = np.zeros_like(rewards, dtype=np.float32)

        enough_steps = steps + 1 >= self.steps_per_assign
        if enough_steps:
            all_ready = all(
                len(self.traj_per_agent[b][ag]) == self.steps_per_assign
                for b in range(self.batch_size)
                for ag in range(self.num_agents)
            )

            if all_ready:
                traj_flat = np.array(
                    [
                        list(self.traj_per_agent[b][ag])
                        for b in range(self.batch_size)
                        for ag in range(self.num_agents)
                    ]
                )  # (batch * agents, steps_per_assign, H, W, C)

                skills_flat = self.current_skills.reshape(-1)  # (batch * agents,)

                ir_flat = self.hsd.compute_intrinsic_reward(traj_flat, skills_flat)
                ir_flat = np.asarray(ir_flat)

                if ir_flat.shape != (self.batch_size * self.num_agents,):
                    raise ValueError(
                        f"[update_buffer] Expected intrinsic reward flat shape "
                        f"({self.batch_size * self.num_agents},), got {ir_flat.shape}"
                    )

                self.intrinsic_rewards = ir_flat.reshape(
                    self.batch_size, self.num_agents
                )

        # Optional scaling of intrinsic reward
        self.intrinsic_rewards *= 0.1

        # ---------------------------------------------------
        # 6) Low-level reward: mix extrinsic & intrinsic
        # ---------------------------------------------------
        rewards_low = self.alpha * rewards + (1.0 - self.alpha) * self.intrinsic_rewards

        # ---------------------------------------------------
        # 7) Insert transition into low-level buffer
        #     done stored as env-level scalar per rollout
        # ---------------------------------------------------
        self.buf_low.add([obs, actions, rewards_low, self.current_skills, next_obs, dones_env])

        # ---------------------------------------------------
        # 8) Update cumulative high-level rewards (macro-step reward)
        #     Use scalar env reward per rollout (average over agents).
        # ---------------------------------------------------
        global_rewards = rewards.mean(axis=1)  # (batch,) # FIXME: why use mean reward over agents?
        self.rewards_high += global_rewards

        # ---------------------------------------------------
        # 9) End of one skill period? -> push high-level transition
        # ---------------------------------------------------
        is_end_of_skill = (steps + 1) % self.steps_per_assign == 0 and steps != 0
        flush_high_level = is_end_of_skill or np.any(dones_env > 0.5)

        if flush_high_level:
            # Cache aggregated reward for logging before it gets reset
            self.episode_high_level_rewards.append(float(np.mean(self.rewards_high)))  # FIXME: mean over recent skill rewards

            # High-level transition uses env-level reward and done
            self.buf_high.add(
                [
                    self.obs_h,          # high-level state at skill start
                    self.share_obs_h,    # shared state
                    self.current_skills,  # high-level action (skills)
                    self.rewards_high,   # accumulated reward over this skill period
                    next_obs,            # next high-level state
                    next_share_obs,
                    dones_env,           # env-level done per rollout
                ]
            )

            # Append trajectories to decoder dataset (train_decoder pads if needed)
            for b in range(self.batch_size):
                for ag in range(self.num_agents):
                    traj_slice = np.array(self.traj_per_agent[b][ag])
                    skill_id = self.current_skills[b][ag]
                    self.dataset.append([traj_slice, skill_id])

            # Reset cumulative rewards; if only some envs ended, keep others accumulating
            if is_end_of_skill:
                self.rewards_high = np.zeros_like(self.rewards_high, dtype=np.float32)
            else:
                done_mask = dones_env > 0.5
                self.rewards_high[done_mask] = 0.0
                # Clear trajectories for finished envs to avoid bleeding across episodes
                for b, done_flag in enumerate(done_mask):
                    if done_flag:
                        for ag in range(self.num_agents):
                            self.traj_per_agent[b][ag].clear()

        # ---------------------------------------------------
        # 10) Advance global step counter
        # ---------------------------------------------------
        self.total_env_steps += 1

    # Fetch low level actions during training mode,
    # manages internal buffers, skill assignments, intrinsic rewards, high level rewards ...
    @torch.no_grad()
    def get_actions_algorithm(self, steps, obs, share_obs, available_actions):  # step within episode
        """
        Compute low-level actions for each agent given current skills.
        Handles:
        - skill assignment and state update at skill boundaries
        - low level action computation via HSD policy
        """

        self.prep_rollout()  # eval mode

        # Start of a new episode; reset trackers that span a full episode
        if steps == 0:
            self.episode_high_level_rewards = []
            self.rewards_high = np.zeros_like(self.rewards_high, dtype=np.float32)

        # ---------------------------------------
        # 1) Validate batch size
        # ---------------------------------------
        incoming_batch = obs.shape[0]
        if incoming_batch != self.batch_size:
            raise ValueError(
                f"[get_actions_algorithm] Incoming batch {incoming_batch} != trainer.batch_size {self.batch_size}"
            )

        # ---------------------------------------
        # 2) Compute low-level actions from HSD policy
        # ---------------------------------------
        raw_actions = self.hsd.get_actions_algorithm(
            steps,
            obs,
            share_obs,
            available_actions,
            self.epsilon,
        )  # (batch, agents, 1)

        actions = raw_actions.squeeze(-1)  # (batch, agents) and include exploration noise

        # ---------------------------------------
        # 3) Skill assignment at boundary
        # ---------------------------------------
        is_skill_boundary = (steps % self.steps_per_assign == 0)

        if is_skill_boundary:
            # save high-level observation snapshot
            self.obs_h = obs
            self.share_obs_h = share_obs

            # use the skills predicted internally by HSD (includes exploration)
            self.current_skills = np.copy(self.hsd.current_skills)

        # ---------------------------------------
        # 4) Return action in env-consumable format
        # ---------------------------------------
        return self._format_actions_for_env(actions)

    # Reset internal variables and storage at the before episode starts again (triggered if batch size changes)
    @torch.no_grad()
    def reset_internals(self, batch_size):
        self.batch_size = batch_size
        self.current_skills = np.zeros((self.batch_size, self.num_agents), dtype=int)
        self.obs_h = None
        self.intrinsic_rewards = np.zeros((self.batch_size, self.num_agents))
        self.traj_per_agent = [
            [deque(maxlen=self.steps_per_assign) for _ in range(self.num_agents)]
            for _ in range(self.batch_size)
        ]
        self.rewards_high = np.zeros((self.batch_size,), dtype=float)
        self.dataset = []
        self.episode_high_level_rewards = []
        self.total_env_steps = 0
        self.next_train_step = self.pretrain_episodes

    @torch.no_grad()
    def prep_rollout(self):
        self.hsd.prep_rollout()

    @torch.no_grad()
    def prep_training(self):
        self.hsd.prep_training()

    @torch.no_grad()
    def save(self, step, save_path: str) -> None:
        """Save HMARL policy to the given path."""
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, f"model_{step}.pt")
        self.hsd.save(model_path)

    @staticmethod
    def _format_actions_for_env(actions: np.ndarray) -> np.ndarray:
        """
        Env expects each action entry to be indexable (a[0]); wrap scalar actions with a
        trailing singleton dimension.
        """
        return np.expand_dims(actions, axis=-1)  # (..., num_agents, 1)


# Single-Agent Wrapper around HMARLTrainer
class HMARLTrainer_PerAgent(HMARLTrainer):
    """
    Single-agent wrapper around HMARLTrainer.
    Behaves identically but accepts per-agent inputs (no agent dimension) and
    internally expands them to match the base multi-agent shapes.
    """

    def __init__(self, config, device=torch.device("cpu")):
        cfg_single = copy.deepcopy(config)
        cfg_single["model"]["num_agents"] = 1
        super().__init__(cfg_single, device=device)

    @torch.no_grad()
    def get_actions_algorithm(self, steps, obs, share_obs, available_actions):
        obs_exp = np.expand_dims(obs, axis=1)
        share_exp = np.expand_dims(share_obs, axis=1) if share_obs is not None else None
        avail_exp = np.expand_dims(available_actions, axis=1)

        actions = super().get_actions_algorithm(steps, obs_exp, share_exp, avail_exp)
        return np.squeeze(actions, axis=1)

    @torch.no_grad()
    def update_buffer(self, steps, obs, share_obs, actions, rewards, next_obs, next_share_obs, dones):
        actions_arr = np.asarray(actions)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr[:, None]
        if actions_arr.ndim > 2 and actions_arr.shape[-1] == 1:
            actions_arr = actions_arr.squeeze(-1)
        if actions_arr.ndim == 2 and actions_arr.shape[1] != 1:
            actions_arr = actions_arr[:, None]

        rewards_arr = np.asarray(rewards)
        if rewards_arr.ndim == 1:
            rewards_arr = rewards_arr[:, None]
        if rewards_arr.ndim == 3 and rewards_arr.shape[-1] == 1:
            rewards_arr = rewards_arr.squeeze(-1)
        if rewards_arr.ndim == 2 and rewards_arr.shape[1] != 1:
            rewards_arr = rewards_arr[:, None]

        dones_arr = np.asarray(dones)
        if dones_arr.ndim == 1:
            dones_arr = dones_arr[:, None]
        if dones_arr.ndim == 3 and dones_arr.shape[-1] == 1:
            dones_arr = dones_arr.squeeze(-1)
        if dones_arr.ndim == 2 and dones_arr.shape[1] != 1:
            dones_arr = dones_arr[:, None]

        return super().update_buffer(
            steps,
            np.expand_dims(obs, axis=1),
            np.expand_dims(share_obs, axis=1) if share_obs is not None else None,
            actions_arr,
            rewards_arr,
            np.expand_dims(next_obs, axis=1),
            np.expand_dims(next_share_obs, axis=1) if next_share_obs is not None else None,
            dones_arr,
        )


# FrozenTrainer Class that wraps pretrained HMARL policies
class FrozenTrainer:
    """Minimal trainer shim for fixed opponent policies."""

    def __init__(self, policy):
        self.policy = policy

    def prep_rollout(self):
        if hasattr(self.policy, "prep_rollout"):
            self.policy.prep_rollout()

    def update_from_source(self):
        # Opponent stays fixed across training iterations.
        return
