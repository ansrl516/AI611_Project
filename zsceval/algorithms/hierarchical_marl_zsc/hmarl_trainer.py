from __future__ import annotations
from collections import deque

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
        self.epsilon_step = (self.epsilon_start - self.epsilon_end) / float(self.epsilon_div)

        # Reward mixing parameters
        self.alpha = cfg_tr["alpha_start"]
        self.alpha_end = cfg_tr["alpha_end"]
        self.alpha_step = cfg_tr["alpha_step"]
        self.alpha_threshold = cfg_tr["alpha_threshold"]

        # Skills
        self.N_skills = cfg_tr["N_skills"]
        self.steps_per_assign = cfg_tr["steps_per_assign"]
        self.decoder_training_threshold = cfg_tr["decoder_training_threshold"]

        # Load model parameters (cleaned and grouped)
        state_dim = cfg_m["state_dim"]
        num_actions = cfg_m["num_actions"]
        obs_dim = cfg_m["obs_dim"]
        self.num_agents = cfg_m["num_agents"]
        self.num_actions = cfg_m["num_actions"]

        # ---- Build HMARL Policy (unchanged logic) ----
        self.hsd = HMARLModel(cfg_m, device=device)

        # ---- Replay buffers ----
        self.buf_high = Replay_Buffer(size=self.buffer_size)
        self.buf_low = Replay_Buffer(size=self.buffer_size)

        # ---- Internal variables ----
        # Note: batch_size from config is only a default; we re-sync shapes at runtime when inputs carry a different leading batch.

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
        self.rewards_high = np.zeros((self.batch_size, self.num_agents), dtype=float)

    ## --- Core functions that is run only in shared overcookedhmarl runner (overcooked_runner_hmarl.py) --- ##

    # Update Q_low, Q_high, decoder based on internal buffer and counter using internals
    def training_step(self, episode_step): 
        # Set training mode for policy
        self.prep_training()

        # LOW LEVEL UPDATE (frequent)
        do_low_update = (
            episode_step >= self.pretrain_episodes
            and episode_step % self.steps_per_train == 0
        )   
        
        if do_low_update:
            batch_low = self.buf_low.sample_batch(self.batch_size)
            self.hsd.train_policy_low(batch_low)

        # HIGH LEVEL UPDATE (slow)
        do_high_update = (
            episode_step % self.steps_per_train == 0 
            and episode_step >= self.pretrain_episodes
        )

        if do_high_update:
            batch_high = self.buf_high.sample_batch(self.batch_size)
            self.hsd.train_policy_high(batch_high)

        # --- DECODER UPDATE ---
        expected_prob = None
        if len(self.dataset) >= self.decoder_training_threshold:
            expected_prob = self.hsd.train_decoder(self.dataset)
            self.dataset = []

        # --- EPSILON / ALPHA SCHEDULE ---
        if episode_step >= self.pretrain_episodes:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_step)

        if self.alpha < self.alpha_threshold:
            self.alpha = min(self.alpha_threshold, self.alpha + self.alpha_step)

        # --- Logging ---
        train_infos = {
            "epsilon": float(self.epsilon),
            "alpha": float(self.alpha),
            "intrinsic_reward_mean": float(np.mean(self.intrinsic_rewards)),
            "high_level_reward_mean": float(np.mean(self.rewards_high))
        }

        if expected_prob is not None:
            train_infos["decoder_expected_prob"] = float(expected_prob)

        flat_skills = self.current_skills.flatten()
        counts = np.bincount(flat_skills, minlength=self.N_skills)
        train_infos["skill_usage"] = (counts / counts.sum()).tolist()

        return train_infos

    # Update buffer and accumulated high level rewards based on environment step
    @torch.no_grad()
    def update_buffer(self, steps, obs, share_obs, actions, rewards, next_obs, next_share_obs, dones):
        # steps: step within episode                
        # info 딕셔너리 key : 
            # "all_agent_obs" : np.array of shape (n_rollout_threads, num_agents, H, W, C)
            # "share_obs" : np.array of shape (n_rollout_threads, num_agents, H, W, C_share)
            # "available_actions" : np.array of shape (n_rollout_threads, num_agents, num_actions)
            # "rewards": np.array of shape (n_rollout_threads, num_agents, 1)
            # "bad_transition" : bool - whether the transition is a bad transition
            # "episode" : dict - episode information
            # "sparse_reward_by_agent" : list of float - sparse reward of the episode by agent - x
            # "shaped_reward_by_agent" : list of float - shaped reward of the episode by agent - x
            # "stuck": list of list of bool - whether the agent is stuck
        
        # Infer effective batch size from obs shape (if no rollout dim, treat as single env with batch_size=0).
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
        # 2) Normalize rewards shape: remove dummy last dim
        #    Expect final shape: (batch_size, num_agents)
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
        # 3) Initialize intrinsic rewards (same shape as rewards)
        # ---------------------------------------------------
        self.intrinsic_rewards = np.zeros_like(rewards, dtype=np.float32)

        # ---------------------------------------------------
        # 4) Compute intrinsic rewards if enough history
        #    Condition: 모든 agent가 최소 steps_per_assign 만큼 trajectory를 모았을 때
        #    (deque maxlen 때문에, 길이가 부족하면 len < steps_per_assign)
        # ---------------------------------------------------
        # We check only when it's even possible:
        # steps is 0-based, so after steps >= steps_per_assign-1 we can have full window.
        enough_steps = steps + 1 >= self.steps_per_assign

        if enough_steps:
            # Check that every deque actually has enough entries
            all_ready = all(
                len(self.traj_per_agent[b][ag]) == self.steps_per_assign
                for b in range(self.batch_size)
                for ag in range(self.num_agents)
            )

            if all_ready:
                # Flatten (batch, agents) -> (batch * agents)
                traj_flat = np.array(
                    [
                        list(self.traj_per_agent[b][ag])  # deque -> list
                        for b in range(self.batch_size)
                        for ag in range(self.num_agents)
                    ]
                )  # shape: (batch * agents, steps_per_assign, ...)

                skills_flat = self.current_skills.reshape(-1)  # (batch * agents,)

                # hsd.compute_intrinsic_reward should return (batch * agents,)
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
            else:
                # 아직 history가 부족한 경우 intrinsic은 0 유지
                pass

        # ---------------------------------------------------
        # 5) Low-level reward: mix extrinsic & intrinsic
        #    rewards_low = alpha * env + (1 - alpha) * intrinsic
        # ---------------------------------------------------
        rewards_low = self.alpha * rewards + (1.0 - self.alpha) * self.intrinsic_rewards

        # ---------------------------------------------------
        # 6) Insert transition into low-level buffer
        # ---------------------------------------------------
        self.buf_low.add(
            [obs, actions, rewards_low, self.current_skills, next_obs, dones]
        )

        # ---------------------------------------------------
        # 7) Update cumulative high-level rewards
        #    Each skill period acts like "one step" in high-level MDP.
        # ---------------------------------------------------
        discounted = self.hsd.gamma ** self.steps_per_assign
        self.rewards_high += rewards * discounted  # shape-wise OK (batch, agents)

        # ---------------------------------------------------
        # 8) End of one skill period? -> push high-level transition
        #    Condition: (steps + 1) % steps_per_assign == 0 and not first step
        # ---------------------------------------------------
        is_end_of_skill = (steps + 1) % self.steps_per_assign == 0 and steps != 0

        if is_end_of_skill:
            # 8-1) Add high-level transition to buffer
            self.buf_high.add(
                [
                    self.obs_h,          # high-level state at skill start
                    self.share_obs_h,    # shared state if any
                    self.current_skills, # high-level action (skills)
                    self.rewards_high,   # accumulated discounted reward over this skill period
                    next_obs,            # next high-level state
                    next_share_obs,
                    dones,
                ]
            )

            # 8-2) Append trajectories to decoder dataset
            #      Each entry: (single-agent trajectory, skill id)
            for b in range(self.batch_size):
                for ag in range(self.num_agents):
                    traj_slice = np.array(self.traj_per_agent[b][ag])  # length == steps_per_assign
                    skill_id = self.current_skills[b][ag]
                    # print("traj_slice.shape in update_buffer:", traj_slice.shape)
                    # print("skill_id in update_buffer:", skill_id)
                    self.dataset.append([traj_slice, skill_id])

            # 8-3) Reset only rewards_high, not traj_per_agent (deque keeps sliding window)
            self.rewards_high = np.zeros_like(self.rewards_high, dtype=np.float32)

            # Note: traj_per_agent is NOT reset. Deque keeps last steps_per_assign frames automatically.

        # ---------------------------------------------------
        # 9) Update per-agent sliding window trajectories
        #    Always push current obs; deque(maxlen) will drop oldest automatically.
        # ---------------------------------------------------
        for b in range(self.batch_size):
            for ag in range(self.num_agents):
                self.traj_per_agent[b][ag].append(obs[b][ag])

    # Fetch low level actions during training mode, 
    # manages internal buffers, skill assignments, intrinsic rewards, high level rewards ... 
    @torch.no_grad()
    def get_actions_algorithm(self, steps, obs, share_obs, available_actions): # step within episode
        """
        Compute low-level actions for each agent given current skills.
        Handles:
        - skill assignment and state update at skill boundaries
        - low level action computation via HSD policy
        """

        self.prep_rollout()   # eval mode

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
        # hsd.get_actions_algorithm returns shape (batch, agents, 1)
        raw_actions = self.hsd.get_actions_algorithm(
            steps,
            obs,
            share_obs,
            available_actions,
            self.epsilon,
        )

        # collapse dummy dim: (batch, agents, 1) → (batch, agents)
        actions = raw_actions.squeeze(-1)

        # ---------------------------------------
        # 3) Skill assignment at boundary
        # ---------------------------------------
        is_skill_boundary = (steps % self.steps_per_assign == 0)

        if is_skill_boundary:
            # save high-level observation snapshot
            self.obs_h = obs
            self.share_obs_h = share_obs

            # pretrain: assign random skills
            if steps < self.pretrain_episodes:
                self.current_skills = np.random.randint(
                    0, self.N_skills, size=(self.batch_size, self.num_agents)
                )
            else:
                # use the skills predicted internally by HSD
                self.current_skills = np.copy(self.hsd.current_skills)

        # ---------------------------------------
        # 4) Pretraining: override low-level actions ONLY during warmup
        #    (random actions consistent with available_actions)
        # ---------------------------------------
        if steps < self.pretrain_episodes:
            actions = np.zeros((self.batch_size, self.num_agents), dtype=np.int32)
            for b in range(self.batch_size):
                for ag in range(self.num_agents):
                    avail = available_actions[b, ag]
                    if np.any(avail):
                        actions[b, ag] = np.random.choice(np.where(avail == 1)[0])
                    else:
                        actions[b, ag] = np.random.randint(self.num_actions)

        # ---------------------------------------
        # 5) Return action in env-consumable format
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
            [[] for _ in range(self.num_agents)] for _ in range(self.batch_size)
        ]
        self.rewards_high = np.zeros_like(self.current_skills)
        self.dataset = []

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
        actions = np.expand_dims(actions, axis=-1)  # shape: (..., num_agents, 1)
        return actions