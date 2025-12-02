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

    # --- Core functions that is run only in shared overcookedhmarl runner (overcooked_runner_hmarl.py) --- #

    # Update Q_low, Q_high, decoder based on internal buffer and counter using internals
    def training_step(self, episode_step): 
        # Set training mode for policy
        self.prep_training()

        # At the beginning of training, do nothing
        if episode_step == 0: 
            return {}

        # Update Q-high and Q-low functions every step * steps_per_train excluding pretraining period
        if episode_step % self.steps_per_train == 0 and episode_step >= self.pretrain_episodes:
            # sample batches randomly from high level buffer and use them for updating Q_high
            batch_high = self.buf_high.sample_batch(self.batch_size)
            self.hsd.train_policy_high(batch_high)

            # sample batches randomly from low level buffer and use them for updating Q_low
            batch_low = self.buf_low.sample_batch(self.batch_size)
            self.hsd.train_policy_low(batch_low)

        # Update decoder if enough dataset has been accumulated       
        if len(self.dataset) >= self.decoder_training_threshold:
            expected_prob = self.hsd.train_decoder(self.dataset)
           # Clear dataset
            self.dataset = []

        # Epsilon for exploration is updated also in here
        if episode_step >= self.pretrain_episodes and self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_step

        # Additionally, update exploration rate epsilon and reward coefficient alpha
        if self.alpha < self.alpha_threshold:
            self.alpha += self.alpha_step
            if self.alpha > self.alpha_threshold:
                self.alpha = self.alpha_threshold

        ## Decide what information to log ##
        train_infos = {}

        # Exploration parameters
        train_infos["epsilon"] = self.epsilon
        train_infos["alpha"] = self.alpha

        # Decoder expected probability (if decoder was trained this step)
        if 'expected_prob' in locals():
            train_infos["decoder_expected_prob"] = float(expected_prob)

        # Skill usage statistics (histogram)
        flat_skills = self.current_skills.flatten()

        skill_counts = np.bincount(flat_skills, minlength=self.N_skills)
        train_infos["skill_usage"] = (skill_counts / skill_counts.sum()).tolist()

        # Intrinsic reward diagnostics
        if hasattr(self, "intrinsic_rewards"):
            train_infos["intrinsic_reward_mean"] = float(np.mean(self.intrinsic_rewards))

        # High-level reward diagnostics
        if hasattr(self, "rewards_high"):
            train_infos["high_level_reward_mean"] = float(np.mean(self.rewards_high))
        ## end of log filling ##
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

            # "sparse_reward_by_agent" : list of float - sparse reward of the episode by agent
            # "shaped_reward_by_agent" : list of float - shaped reward of the episode by agent
            # "stuck": list of list of bool - whether the agent is stuck

            ## all_agent_obs, share_obs, sparse_reward_by_agent, shaped_reward_by_agent -> 핵심 정보 
        
        # Infer effective batch size from obs shape (if no rollout dim, treat as single env with batch_size=0).
        obs_np = np.array(obs)
        incoming_batch = obs_np.shape[0]
        assert incoming_batch == self.batch_size
        # if incoming_batch != self.batch_size:
        #     self.reset_internals(incoming_batch)
        #     return

        # remove last dim of rewards because it is dummy
        rewards = rewards.squeeze(-1)  # shape: (batch_size, num_agents)

        # Update low-level buffer with intrinsic reward and store it into buffer
        self.intrinsic_rewards = np.zeros((self.batch_size, self.num_agents))

        if steps > self.steps_per_assign:
            # Flatten agent and batch dims → compute intrinsic rewards in one shot
            traj_flat = np.array(
                [
                    self.traj_per_agent[batch_idx][idx_agent][-self.steps_per_assign:]
                    for batch_idx in range(self.batch_size)
                    for idx_agent in range(self.num_agents)
                ]
            )  # shape: (num_agents * batch_size, steps_per_assign, H, W, C)
            skills_flat = self.current_skills.reshape(-1)  # batch-major flatten
            ir_flat = self.hsd.compute_intrinsic_reward(traj_flat, skills_flat)  # shape: (num_agents * batch_size,)
            self.intrinsic_rewards = ir_flat.reshape(self.batch_size, self.num_agents)  # back to (batch, agents)
        else:
            # Not enough history yet; keep intrinsic rewards at zero until trajectories accumulate.
            self.intrinsic_rewards[:] = 0.0

        print("rewards_shape:", rewards.shape)
        print("intrinsic_rewards_shape:", self.intrinsic_rewards.shape)
        rewards_low = self.alpha * rewards + (1 - self.alpha) * self.intrinsic_rewards

        self.buf_low.add([obs, actions, rewards_low, self.current_skills, next_obs, dones])
        
        # Update accumulated rewards for high level policy
        self.rewards_high += rewards * (self.hsd.gamma**self.steps_per_assign)

        # Update high-level buffer and then reset accumulated high level rewards at the end of this skill assignment period
        if (steps+1) % self.steps_per_assign == 0 and steps != 0:
            self.buf_high.add([self.obs_h, self.share_obs_h, self.current_skills, self.rewards_high, next_obs, next_share_obs, dones])
            
            # Append skill-trajectory to dataset for training decoder (regardless of agent and batch)
            # note if size of dataset exceeds threshold, clearing decoder is done in training_step
            for batch_idx in range(self.batch_size):
                for idx_agent in range(self.num_agents):
                    self.dataset.append([self.traj_per_agent[batch_idx][idx_agent][-self.steps_per_assign:], self.current_skills[batch_idx][idx_agent]])
            # Reset accumulated high level rewards
            self.rewards_high = np.zeros_like(self.rewards_high)

            # Reset trajectory per agent data structure
            self.traj_per_agent = [
                [[] for _ in range(self.num_agents)] for _ in range(self.batch_size)
            ]

        # Update trajectory per agent data structure
        for batch_idx in range(self.batch_size):
            for idx_agent in range(self.num_agents):
                self.traj_per_agent[batch_idx][idx_agent].append((obs[batch_idx][idx_agent]))
    # Fetch low level actions during training mode, 
    # manages internal buffers, skill assignments, intrinsic rewards, high level rewards ... 
    @torch.no_grad()
    def get_actions_algorithm(self, steps, obs, share_obs, available_actions): # step within episode
        # info 딕셔너리 key : 
            # "all_agent_obs" : np.array of shape (n_rollout_threads, num_agents, H, W, C)
            # "share_obs" : np.array of shape (n_rollout_threads, num_agents, H, W, C_share)
            # "available_actions" : np.array of shape (n_rollout_threads, num_agents, num_actions)
            # "bad_transition" : bool - whether the transition is a bad transition
            # "episode" : dict - episode information

            # "sparse_reward_by_agent" : list of float - sparse reward of the episode by agent
            # "shaped_reward_by_agent" : list of float - shaped reward of the episode by agent
            # "stuck": list of list of bool - whether the agent is stuck

            ## all_agent_obs, share_obs, sparse_reward_by_agent, shaped_reward_by_agent -> 핵심 정보 


        self.prep_rollout() # set eval mode for policy

        
        incoming_batch = obs.shape[0] 
        print("batch_size_comparison ",incoming_batch, self.batch_size)
        assert incoming_batch == self.batch_size

        # Trainer internally includes policy, and updates buffers, current skills, intrinsic rewards, ... internally
        # so it only prints out actions, actual training algorithm of hmarl is hidden
        # skills_int is the newly assigned skills at this step (or is just the same skill within period)
        actions = self.hsd.get_actions_algorithm(
            steps,
            obs,
            share_obs,
            available_actions,
            self.epsilon,
        ).squeeze(-1) # shape: [batch_size, num_agents] where we collapse dummy dim 1
        print("actions_shape_from_hsd ", actions.shape)
        skills_int = self.hsd.current_skills

        # At time of new skill assignment, update new skill and store it to current_skills
        if steps % self.steps_per_assign == 0:
            self.obs_h = obs
            self.share_obs_h = share_obs

            # Update new skills (balance between exploration and exploitation)
            if steps < self.pretrain_episodes: # random skill assignment during warmup
                skills_int = np.random.randint(0, self.N_skills, self.num_agents) if self.batch_size == 0 \
                    else np.random.randint(0, self.N_skills, (self.batch_size, self.num_agents))

            self.current_skills = skills_int

        # Balance between exploitation and exploration (exploration decay control is done at training_step)
        # Select random actions for each agent from available_actions   
        # Note available_actions is expected to be a mask array of shape (num_agents, num_actions) or (batch_size, num_agents, num_actions)
        if steps < self.pretrain_episodes:
            actions = np.zeros((self.batch_size, self.num_agents), dtype=np.int32)

            for b in range(self.batch_size):
                for agent in range(self.num_agents):
                    avail = available_actions[b, agent]   # shape (num_actions,)
                    if np.any(avail):
                        actions[b, agent] = np.random.choice(np.where(avail == 1)[0])
                    else:
                        actions[b, agent] = np.random.randint(self.num_actions)

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