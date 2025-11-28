"""Overcooked environment wrapper exposing the interface expected by train_hsd.py."""

import types
from pathlib import Path

import numpy as np

from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as OvercookedNew


class _Args(types.SimpleNamespace):
    """Minimal argparse-like container for Overcooked constructor."""


class Env:
    def __init__(self, config_env, config_main, test=False, N_roles=None):
        # Map provided config to Overcooked args; fall back to sensible defaults
        args = _Args()
        args.num_agents = config_env.get("num_home_players", 2)
        args.layout_name = config_env.get("layout_name", "cramped_room")
        args.episode_length = config_env.get("max_steps", 400)
        args.use_render = config_main.get("render", False)
        args.initial_reward_shaping_factor = config_env.get("initial_reward_shaping_factor", 1.0)
        args.reward_shaping_factor = config_env.get("reward_shaping_factor", 1.0)
        args.reward_shaping_horizon = config_env.get("reward_shaping_horizon", 2.5e6)
        args.use_phi = False
        args.use_hsp = False
        args.random_index = config_env.get("random_index", False)
        args.random_start_prob = config_env.get("random_start_prob", 0.0)
        args.use_available_actions = True
        args.n_render_rollout_threads = 1
        args.overcooked_version = config_env.get("overcooked_version", "new")
        args.use_agent_policy_id = False
        args.use_identity_feature = False
        args.use_timestep_feature = False
        args.use_agent_policy_id = False
        args.use_random_terrain_state = False
        args.use_random_player_pos = False
        args.agent_policy_names = ["ppo"] * args.num_agents
        args.overcooked_version = "new"
        args.use_random_terrain_state = False
        args.use_random_player_pos = False
        args.num_initial_state = 1

        self.env = OvercookedNew(all_args=args, run_dir=Path("."))
        self.N_home = args.num_agents
        self.N_roles = N_roles
        self.test = test
        self.env_step = 0
        self.control_team = 0
        self.control_index = 0

        # Peek at observation/action dimensions
        obs_batch, info = self.env.reset()
        all_agent_obs = np.array(info["all_agent_obs"])
        share_obs = np.array(info["share_obs"])
        self.obs_dim = all_agent_obs.reshape(self.N_home, -1).shape[1]
        self.state_dim = share_obs.reshape(self.N_home, -1).shape[1]
        self.action_dim = self.env.action_space[0].n

    def random_actions(self):
        return np.random.randint(0, self.action_dim, size=self.N_home)

    def reset(self):
        self.env_step = 0
        obs_batch, info = self.env.reset()
        return self._format_obs(info)

    def step(self, actions_int):
        obs, share_obs, rewards, dones, infos, avail_actions = self.env.step(actions_int)
        self.env_step += 1

        state_home, state_away, list_obs_home, list_obs_away, done = self._format_obs(infos)

        # Use mean reward as team reward
        reward = float(np.mean(rewards))
        local_rewards = np.array(rewards).flatten()

        info = {"winning_team": 0 if reward > 0 else -1}
        done_flag = bool(np.any(dones))
        return state_home, state_away, list_obs_home, list_obs_away, reward, local_rewards, done_flag, info

    def _format_obs(self, info):
        all_agent_obs = np.array(info["all_agent_obs"])
        share_obs = np.array(info["share_obs"])

        list_obs_home = all_agent_obs.reshape(self.N_home, -1)
        list_obs_away = np.zeros_like(list_obs_home)
        state_home = share_obs.reshape(self.N_home, -1)
        state_away = np.zeros_like(state_home)
        done = False
        return state_home, state_away, list_obs_home, list_obs_away, done
