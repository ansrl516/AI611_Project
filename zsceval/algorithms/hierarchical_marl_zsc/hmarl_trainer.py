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

# ALG_DIR = Path(__file__).parent / "alg"
# RESULTS_DIR = ALG_DIR.parent / "results"
# DEFAULT_SAVE_DIR = Path(__file__).parent / "saved"
# DEFAULT_CONFIG = ALG_DIR / "config.json"


# def _ensure_alg_path() -> None:
#     """Make sure alg/ is importable."""
#     alg_path = str(ALG_DIR.resolve())
#     if alg_path not in sys.path:
#         sys.path.append(alg_path)


# def _load_config(config_path: Optional[Path] = None) -> dict:
#     cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG
#     with open(cfg_path, "r") as f:
#         return json.load(f)


# @dataclass(frozen=True)
# class EnvSpec:
#     """Minimal environment specification needed to construct HSD networks."""

#     num_agents: int
#     state_dim: int
#     obs_dim: int
#     num_actions: int


# class HMARLAgent:
#     """Inference-only wrapper around the HSD algorithm."""

#     def __init__(self, alg, epsilon: float = 0.0):
#         self.alg = alg
#         self.epsilon = epsilon

#     @classmethod
#     def from_checkpoint(
#         cls,
#         checkpoint: Path,
#         env_spec: EnvSpec,
#         config_path: Optional[Path] = None,
#         device=None,
#     ) -> "HMARLAgent":
#         """Load a saved HSD checkpoint and return a ready-to-use agent."""
#         _ensure_alg_path()
#         import alg_hsd  # type: ignore

#         config = _load_config(config_path)
#         alg_cfg = config["alg"]
#         h_cfg = config["h_params"]
#         nn_cfg = config["nn_hsd"]

#         alg = alg_hsd.Alg(
#             alg_cfg,
#             h_cfg,
#             env_spec.num_agents,
#             env_spec.state_dim,
#             env_spec.obs_dim,
#             env_spec.num_actions,
#             h_cfg["N_skills"],
#             nn_cfg,
#             device=device,
#         )
#         alg.load(str(checkpoint))
#         return cls(alg)

#     def assign_skills(self, observations: Sequence, n_skills_current: Optional[int] = None, epsilon: Optional[float] = None):
#         """Choose a skill for each agent using the high-level policy."""
#         n_skills = n_skills_current or getattr(self.alg, "l_z", None)
#         if n_skills is None:
#             raise ValueError("Unable to infer number of skills; pass n_skills_current explicitly.")
#         eps = self.epsilon if epsilon is None else epsilon
#         return self.alg.assign_skills(observations, eps, n_skills)

#     def act(self, observations: Sequence, skills_int, epsilon: Optional[float] = None):
#         """Select primitive actions for all agents given current skills."""
#         eps = self.epsilon if epsilon is None else epsilon
#         return self.alg.run_actor(observations, skills_int, eps)


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
        
        # config for curriculum learning (Number of skills increase gradually)
        self.N_skills_current = config_h["N_skills_start"] 
        assert self.N_skills_current <= self.N_skills
        self.curriculum_threshold = config_h["curriculum_threshold"]

        # config for reward coefficient
        self.alpha = config_h["alpha_start"]
        self.alpha_end = config_h["alpha_end"]
        self.alpha_step = config_h["alpha_step"]
        self.alpha_threshold = config_h["alpha_threshold"]

        # config for Number of single-agent trajectory segments used for each decoder training step
        self.decoder_training_threshold = config_h["N_batch_hsd"]

        # We will later remove env storage
        self.env = base_env # instance of ZSC-Eval env  

        state_dim = self.env.state_dim
        num_actions = self.env.num_actions
        obs_dim = self.env.obs_dim
        num_agents = self.env.num_agents  # number of agents

        # Import Policy and Trainer
        self.hsd = HMARLModel(config_param_sharing_option, config_main, config_h, num_agents, state_dim, obs_dim, num_actions, self.N_skills, config["nn_hsd"])
        # To reuse some of the trajectories, we maintain high & low level replay buffers:
        self.buf_high = replay_buffer.Replay_Buffer(size=self.buffer_size)
        self.buf_low = replay_buffer.Replay_Buffer(size=self.buffer_size)

        # Dataset of [obs traj, skill] for training decoder
        self.dataset = []

    def train(self):
        expected_prob = 0
        step = 0
        step_train = 0
        step_h = 0
        
        for idx_episode in range(1, self.N_train+1):
            # Get initial information from the zsc env instance
            policy_obs, share_obs, available_actions = self.env.reset()
            done = False
            # Update high-level state by storing state & obs at this timepoint
            policy_obs_high, share_obs_high = policy_obs, share_obs
            # Use cumulative discounted reward for high level policy
            rewards_high = np.zeros(self.num_agents)
            # Distributes random skills at start of the episode
            skills_int = np.random.randint(0, self.N_skills_current, self.num_agents)
            # Data structure for storing trajectories per agent, used for computing intrinsic rewards (skill decode prob)
            traj_per_agent = [[] for _ in range(self.num_agents)] # stores trajectory of each agent up to steps_per_assign
            # Loop for each episode
            step_episode = 0 # steps within an episode

            while not done:
                # Data structure for storing intrinsic reward per agent
                intrinsic_rewards = np.zeros(self.num_agents)

                # High-level calculations at every steps_per_assign steps
                if step_episode % self.steps_per_assign == 0:
                    # Update high-level buffers
                    if step_episode != 0:
                        # Store data for high level policy training 
                        rewards_high = rewards_high * (self.config_alg['gamma']**self.steps_per_assign) # FIXME
                        self.buf_high.add([share_obs_high, policy_obs_high, skills_int, rewards_high, share_obs, policy_obs, done])

                        # Compute intrinsic rewards for each agent based on decoder prediction <- low level policy에 들어가야 함 (FIXME)
                        for idx_agent in range(self.num_agents):
                            traj = np.array(traj_per_agent[idx_agent][-self.steps_per_assign:])  # shape [obs_dim]
                            intrinsic_rewards[idx_agent] = self.hsd.compute_intrinsic_reward(
                                traj, skills_int[idx_agent]
                            )
                        
                        # Store skill-trajectory for training decoder
                        for idx_agent in range(self.num_agents):
                            self.dataset.append([traj_per_agent[idx_agent][-self.steps_per_assign:], skills_int[idx_agent]])

                    # Decide new skills for all agents
                    if idx_episode < self.pretrain_episodes: # random skill assignment during warmup
                        skills_int = np.random.randint(0, self.N_skills_current, self.num_agents)
                    else:
                        skills_int = self.hsd.assign_skills(share_obs, self.N_skills_current, self.epsilon)
                    
                    # Update high-level state by storing state & obs at this timepoint
                    share_obs_high, policy_obs_high = share_obs, policy_obs

                    # Reset cumulative discounted reward for high level policy
                    rewards_high = np.zeros_like(rewards_high)

                    # update high-level actions every step_h * steps_per_train
                    if (idx_episode >= self.pretrain_episodes) and (step_h % self.steps_per_train == 0):
                        # sample batches randomly from high level buffer and use them for updating Q_high
                        batch = self.buf_high.sample_batch(self.batch_size)
                        self.hsd.train_policy_high(batch)

                        step_train += 1
                    
                    # udpate number of high level steps
                    step_h += 1
                    
                # Low-level calculations conditioned on high level skill assignment
                if idx_episode < self.pretrain_episodes:
                    # Select random actions for each agent from available_actions
                    # Note available_actions is expected to be a mask array of shape (num_agents, num_actions)
                    actions_int = np.array([
                        np.random.choice(np.where(avail == 1)[0])
                        if np.any(avail)
                        else np.random.randint(available_actions.shape[1])
                        for avail in available_actions
                    ])
                else: # use low-level policy to select actions - tuples agent1 action, ... agentN action
                    actions_int = self.hsd.get_actions(policy_obs, skills_int, self.epsilon) # shape: [num_agents,] - 0 ~ 5 per agent
                
                # Perform low-level step in environment (TODO: keep checking Overcooked_Env.py)
                policy_obs_next, share_obs_next, policy_rewards, done, infos, available_actions = self.env.step(actions_int) # shape: 경윤님 채워주세요! - info dictionary 안에 policy_obs_next, ...

                # Compute low level rewards using intrisic_reward for each agent
                rewards_low = policy_rewards
                rewards_low = self.alpha * rewards_low + (1 - self.alpha) * intrinsic_rewards

                # Update low-level buffer and then move to next state
                self.buf_low.add([policy_obs, actions_int, rewards_low, skills_int, policy_obs_next, done])
                policy_obs = policy_obs_next
                share_obs = share_obs_next

                # Store it into trajectory
                for idx_agent in range(self.num_agents):
                    traj_per_agent[idx_agent].append((policy_obs[idx_agent], actions_int[idx_agent], rewards_low[idx_agent])) # FIXME

                # update low-level policies every step * steps_per_train
                if (idx_episode >= self.pretrain_episodes) and (step % self.steps_per_train == 0):
                    # sample batches randomly from high level buffer and use them for updating Q_low
                    batch = self.buf_low.sample_batch(self.batch_size)
                    self.hsd.train_policy_low(batch)
                    step_train += 1

                # Update reward_high
                rewards_high += policy_rewards 

                # Update step_episode
                step += 1

            # Episode is done, terminate the current skill assignment and do the same operation
            # 1. Store buf_high 2. Store trajectory
            rewards_high_ = rewards_high * (self.config_alg['gamma']**self.steps_per_assign) # FIXME
            self.buf_high.add([share_obs_high, policy_obs_high, skills_int, rewards_high_, done])
            if step_episode >= self.steps_per_assign:
                for idx_agent in range(self.num_agents):
                    self.dataset.append([traj_per_agent[idx_agent][-self.steps_per_assign:], skills_int[idx_agent]])

            # If dataset has accumulated enough, train decoder
            if len(self.dataset) >= self.decoder_training_threshold:
                expected_prob = self.hsd.train_decoder(self.dataset)
                # If expected_prob is high, then increase number of skills (FIXME: Delete)
                if expected_prob > self.curriculum_threshold:
                    self.N_skills_current = min(int(1.5 * self.N_skills_current + 1), self.N_skills)
                # Clear dataset
                self.dataset = []
                step_train +=1

            # If randomness is enough (FIXME: Fix the comment)
            if idx_episode >= self.pretrain_episodes and self.epsilon > self.epsilon_end:
                self.epsilon -= self.epsilon_step
            
            # Logging, Saving, Evaluating (TODO)


    # Later add entrypoint for pretraine decoder, low level policy decision, ...
    @torch.no_grad()
    def act_in_pretrained(self, steps, env_msg): # env unbatched version
        """
        Pretrained inference-only hierarchical policy:
        - Internally updates and maintains skill assignments according to steps per batch
        - Takes env_msg from ZSC-Eval env instance and returns primitive actions for all agents

        Shape of env_msg: [policy_obs, share_obs, available_actions]
            - policy_obs: list of np.array of shape (num_agents, obs_dim) for each agent
            - share_obs: np.array of shape (num_agents, state_dim)
            - available_actions: np.array of shape (num_agents, num_actions) as mask of available actions

        Algorithm:
        - Every `steps_per_assign` steps:
              Update skills using the high-level policy and store it internally

        - Every step:
              Compute actions using the low-level policy using stored skills

        Returns:
            actions_int : np.array of shape (num_agents,)
        """

        policy_obs, share_obs, available_actions = env_msg
        N = self.num_agents

        # 1. Initialize stored skills on first call
        if not hasattr(self, "_current_skills"):
            self._current_skills = np.random.randint(
                0, self.N_skills_current, size=N
            )

        # 2. High-level update every steps_per_assign
        if steps % self.steps_per_assign == 0:
            self._current_skills = self.hsd.assign_skills(
                share_obs,
                self.N_skills_current,
                epsilon=self.epsilon
            )

        # 3. Low-level action selection
        actions_int = self.hsd.get_actions(
            policy_obs,
            self._current_skills,
            available_actions,
            epsilon=self.epsilon
        )

        return actions_int
    
    @torch.no_grad()
    def reset(self):
        """Reset internal skill storage."""
        if hasattr(self, "_current_skills"):
            del self._current_skills
        if hasattr(self, "_current_skills_batch"):
            del self._current_skills_batch

    @torch.no_grad()
    def act_in_pretrained_batch(self, steps, env_msg):
        """
        Batched version of act_in_pretrained.

        Args:
            steps : int
            env_msg = (batch_policy_obs, batch_share_obs, batch_available_actions)
            batch_policy_obs:        [B, N, obs_dim]
            batch_share_obs:         [B, N, state_dim]
            batch_available_actions: [B, N, num_actions]

        Returns:
            actions_int_batch: np.array [B, N]
        """

        batch_policy_obs, batch_share_obs, batch_available_actions = env_msg
        B, N, _ = batch_policy_obs.shape

        # 1. Initialize persistent batch skill storage on first call
        if not hasattr(self, "_current_skills_batch"):
            self._current_skills_batch = np.random.randint(
                0, self.N_skills_current, size=(B, N)
            )

        # 2. High-level skill update every steps_per_assign
        if steps % self.steps_per_assign == 0:
            self._current_skills_batch = self.hsd.assign_skills_batch(
                batch_share_obs,
                N_skills_current=self.N_skills_current,
                epsilon=self.epsilon
            )

        # 3. Low-level action selection
        actions_int_batch = self.hsd.get_actions_batch(
            batch_policy_obs,
            self._current_skills_batch,
            batch_available_actions,
            epsilon=self.epsilon
        )

        return actions_int_batch