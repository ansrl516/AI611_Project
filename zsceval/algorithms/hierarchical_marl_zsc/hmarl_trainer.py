from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from hmarl_policy import HMARLModel
import utils.replay_buffer as replay_buffer

# Trainer Class Compatible with ZSC-Eval
# In training, it is wrapped with simplest runner which not compatible with base_runner
# After training, it provides functions other policies can use (decoder, assign_skills, get_actions, ...)
#                 it has function which creates fixed Agent Instances
class HMARLTrainer:
    """Wrapper to bridge ZSC env messaging with HMARL policy/trainer."""

    def __init__(self, config, base_env):
        # Setup configs
        config_param_sharing_option = config["param_sharing_option"] # decide parameter sharing for Q_low and Q_high
        config_main = config["main"]
        config_alg = config["alg"] # parameter related to general training settings
        config_h = config["h_params"] # important parameter

        seed = config_main["seed"]
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        dir_name = config_main["dir_name"]
        self.save_period = config_main["save_period"]

        os.makedirs("../results/%s" % dir_name, exist_ok=True)
        with open("../results/%s/%s" % (dir_name, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # config for training settings
        self.N_train = config_alg["N_train"]
        self.N_eval = config_alg["N_eval"]
        self.period = config_alg["period"]
        self.buffer_size = config_alg["buffer_size"]
        self.batch_size = config_alg["batch_size"]
        self.pretrain_episodes = config_alg["pretrain_episodes"]
        self.steps_per_train = config_alg["steps_per_train"]

        # config for exploration
        self.epsilon_start = config_alg["epsilon_start"]
        self.epsilon_end = config_alg["epsilon_end"]
        self.epsilon_div = config_alg["epsilon_div"]
        self.epsilon_step = (self.epsilon_start - self.epsilon_end) / float(self.epsilon_div)
        self.epsilon = self.epsilon_start

        # config for skills
        self.N_skills = config_h["N_skills"] # Final number of skills
        self.steps_per_assign = config_h["steps_per_assign"] # Number of steps per skill assignment
        
        # config for reward coefficient
        self.alpha = config_h["alpha_start"]
        self.alpha_end = config_h["alpha_end"]
        self.alpha_step = config_h["alpha_step"]
        self.alpha_threshold = config_h["alpha_threshold"]

        # config for Number of single-agent trajectory segments used for each decoder training step
        self.decoder_training_threshold = config_h["N_batch_hsd"]

        state_dim = self.state_dim # shared_obs for centralized critic
        num_actions = self.num_actions 
        obs_dim = self.obs_dim # for policy input (Q_low, Q_high, decoder)
        num_agents = self.num_agents  # number of agents

        # Import Policy and Trainer
        self.hsd = HMARLModel(config_param_sharing_option, config_main, config_h, num_agents, state_dim, obs_dim, num_actions, self.N_skills, config["nn_hsd"])
        
        ## From here, setup internal buffers and counters for training ##
        # To reuse some of the trajectories for updating Qs, we maintain high & low level replay buffers (regardless of agent & batch):
        self.buf_high = replay_buffer.Replay_Buffer(size=self.buffer_size)
        self.buf_low = replay_buffer.Replay_Buffer(size=self.buffer_size)

        # Other internal variables
        self.current_skills = np.zeros((self.batch_size, self.num_agents), dtype=int) if self.batch_size > 0 else np.zeros(self.num_agents, dtype=int) # current skill assignment for each agent in the batch
        self.obs_h = None # high level observation at the time of skill assignment
        self.intrinsic_rewards = np.zeros((self.batch_size, self.num_agents)) if self.batch_size > 0 else np.zeros(self.num_agents) # intrinsic rewards for each agent in the batch

        # Dataset of [obs traj, skill] for training decoder (regardless of agent and batch)
        self.dataset = []

        # Datastructure for internally storing trajectories of each agent during previous period
        # stores trajectory of each agent up to steps_per_assign
        if self.batch_size == 0:
            self.traj_per_agent = [ [] for _ in range(self.num_agents) ]
        else:
            self.traj_per_agent = [
                [[] for _ in range(self.batch_size)]
                for _ in range(self.num_agents)
            ]

        # Datastructure for High level reward
        self.rewards_high = np.zeros((self.batch_size, self.num_agents)) if self.batch_size > 0 else np.zeros(self.num_agents)

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

        # Decide which information to log (TODO)
        train_infos = {}

        # Additionally, update exploration rate epsilon and reward coefficient alpha
        if self.alpha < self.alpha_threshold:
            self.alpha += self.alpha_step
            if self.alpha > self.alpha_threshold:
                self.alpha = self.alpha_threshold
        if episode_step >= self.pretrain_episodes and self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_step

        return train_infos
    
    # Update buffer and accumulated high level rewards based on environment step
    @torch.no_grad()
    def update_buffer(self, steps, obs, share_obs, actions, rewards, next_obs, next_share_obs, dones):
        # steps: step within episode                
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
        
        # Update low-level buffer with intrinsic reward and store it into buffer
        if self.batch_size > 0:
            intrinsic_rewards = np.zeros((self.batch_size, self.num_agents))
        intrinsic_rewards = np.zeros((self.batch_size, self.num_agents)) if self.batch_size > 0 else np.zeros(self.num_agents)

        for idx_agent in range(self.num_agents):
            traj = np.array(self.traj_per_agent[idx_agent][-self.steps_per_assign:])  # shape [obs_dim]
            intrinsic_rewards[idx_agent] = self.hsd.compute_intrinsic_reward(
                traj, self.current_skills[idx_agent]
            )

        rewards_low = self.alpha * rewards + (1 - self.alpha) * intrinsic_rewards

        self.buf_low.add([obs, actions, rewards_low, self.current_skills, next_obs, dones])
        
        # Update accumulated rewards for high level policy
        self.rewards_high += rewards * (self.config_alg['gamma']**self.steps_per_assign)

        # Update high-level buffer and then reset accumulated high level rewards at the end of this skill assignment period
        if steps % self.steps_per_assign == -1 and steps != 0:
            self.buf_high.add([self.obs_h, self.share_obs_h, self.current_skills, self.rewards_high, next_obs, next_share_obs, dones])
            
            # Reset accumulated high level rewards
            self.rewards_high = np.zeros_like(self.rewards_high)

            # Reset trajectory per agent data structure
            if self.batch_size == 0:
                self.traj_per_agent = [ [] for _ in range(self.num_agents) ]
            else:
                self.traj_per_agent = [
                    [[] for _ in range(self.batch_size)]
                    for _ in range(self.num_agents)
                ]

            # Append skill-trajectory to dataset for training decoder (regardless of agent and batch)
            # note if size of dataset exceeds threshold, clearing decoder is done in training_step
            if self.batch_size == 0:
                for idx_agent in range(self.num_agents):
                    self.dataset.append([self.traj_per_agent[idx_agent][-self.steps_per_assign:], self.current_skills[idx_agent]])
            else:
                for batch_idx in range(self.batch_size):
                    for idx_agent in range(self.num_agents):
                        self.dataset.append([self.traj_per_agent[batch_idx][idx_agent][-self.steps_per_assign:], self.current_skills[batch_idx][idx_agent]])
        
            # Update starting obs of the period
            self.obs_h = next_obs
            self.share_obs_h = next_share_obs

        # Update trajectory per agent data structure
        if self.batch_size == 0:
            for idx_agent in range(self.num_agents):
                self.traj_per_agent[idx_agent].append((obs[idx_agent], actions[idx_agent], rewards[idx_agent]))
        else:
            for batch_idx in range(self.batch_size):
                for idx_agent in range(self.num_agents):
                    self.traj_per_agent[batch_idx][idx_agent].append((obs[batch_idx][idx_agent], actions[batch_idx][idx_agent], rewards[batch_idx][idx_agent]))

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

        # Trainer internally includes policy, and updates buffers, current skills, intrinsic rewards, ... internally
        # so it only prints out actions, actual training algorithm of hmarl is hidden
        # skills_int is the newly assigned skills at this step (or is just the same skill within period)
        actions, skills_int = self.hsd.get_actions_algorithm(steps, obs, share_obs, available_actions, self.epsilon)

        # At time of new skill assignment, update new skill and store it to current_skills
        if steps % self.steps_per_assign == 0 and steps != 0:

            # Update new skills (balance between exploration and exploitation)
            if steps < self.pretrain_episodes: # random skill assignment during warmup
                skills_int = np.random.randint(0, self.N_skills, self.num_agents) if self.batch_size == 0 \
                    else np.random.randint(0, self.N_skills, (self.batch_size, self.num_agents))

            self.current_skills = skills_int

        # Balance between exploitation and exploration (exploration decay control is done at training_step)
        # Select random actions for each agent from available_actions   
        # Note available_actions is expected to be a mask array of shape (num_agents, num_actions) or (batch_size, num_agents, num_actions)
        if steps < self.pretrain_episodes:
            if self.batch_size == 0:
                actions = np.array([
                    np.random.choice(np.where(avail == 1)[0])
                    if np.any(avail)
                    else np.random.randint(self.num_actions)
                    for avail in available_actions
                ])
            
            else:
                actions = np.zeros((self.batch_size, self.num_agents), dtype=np.int32)

                for b in range(self.batch_size):
                    for agent in range(self.num_agents):
                        avail = available_actions[b, agent]   # shape (num_actions,)
                        if np.any(avail):
                            actions[b, agent] = np.random.choice(np.where(avail == 1)[0])
                        else:
                            actions[b, agent] = np.random.randint(self.num_actions)

        return actions # shape: [batch_size, num_agents] or [num_agents,] depending on batch_size with value 0 ~ num_actions-1

    # Reset internal variables at the before episode starts again
    @torch.no_grad()
    def reset_internals(self):
        self.current_skills = np.zeros((self.batch_size, self.num_agents), dtype=int) if self.batch_size > 0 else np.zeros(self.num_agents, dtype=int)
        self.obs_h = None
        self.intrinsic_rewards = np.zeros((self.batch_size, self.num_agents)) if self.batch_size > 0 else np.zeros(self.num_agents)

        if self.batch_size == 0:
            self.traj_per_agent = [ [] for _ in range(self.num_agents) ]
        else:
            self.traj_per_agent = [
                [[] for _ in range(self.batch_size)]
                for _ in range(self.num_agents)
            ]

        self.rewards_high = np.zeros((self.batch_size, self.num_agents)) if self.batch_size > 0 else np.zeros(self.num_agents)

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
        self.hsd.save(step, save_path)