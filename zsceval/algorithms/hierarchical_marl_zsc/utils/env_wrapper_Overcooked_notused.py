import numpy as np


class OvercookedHMARLInterface:
    """
    Minimal environment wrapper to train HSD on Overcooked.
    Only implements the methods used by train_hsd.py.
    """

    def __init__(self, base_env):
        """
        base_env: Overcooked env instance with gym-like API / Env class / Worker from ZSC-Eval:
                  obs, reward, done, info = env.step(action_list)
        num_agents: number of controlled agents (Usually 2)
        """
        self.base_env = base_env
        self.N_home  = self.base_env.num_agents

        # ---- Infer observation dimensions ----
        raw_obs = self._reset_raw()
        list_obs = self._split_obs(raw_obs)
        self.obs_dim = list_obs[0].shape[0]

        # ---- Infer action dimension ----
        self.action_dim = self._infer_action_dim()

        # Global state = concatenation of all agent obs
        self.state_dim = self.obs_dim * self.N_home

    # ===== Internal helpers =====

    def _reset_raw(self):
        out = self.base_env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
        else:
            obs = out
        return obs

    def _split_obs(self, joint_obs):
        """
        joint_obs can be:
            - list/tuple of obs per agent
            - flat array representing combined obs
        """
        if isinstance(joint_obs, (list, tuple)):
            return [np.asarray(o).flatten().astype(np.float32) for o in joint_obs]

        joint_obs = np.asarray(joint_obs).flatten()
        per_agent = np.split(joint_obs, self.N_home)
        return [o.astype(np.float32) for o in per_agent]

    def _infer_action_dim(self):
        space = self.base_env.action_space
        if hasattr(space, "n"):
            return int(space.n)
        if hasattr(space, "nvec"):
            return int(space.nvec[0])
        raise ValueError("Cannot infer discrete action dimension")

    def _format_action(self, actions):
        return list(actions)

    # ===== Required HSD API =====

    def reset(self):
        raw_obs = self._reset_raw()
        list_obs_home = self._split_obs(raw_obs)

        state_home = np.concatenate(list_obs_home, axis=0)
        state_away = None            # HSD ignores this
        list_obs_away = []           # HSD ignores this
        done = False

        return state_home, state_away, list_obs_home, list_obs_away, done

    def step(self, actions_int):
        joint_action = self._format_action(actions_int)
        step_out = self.base_env.step(joint_action)

        # Support Gymnasium output
        if len(step_out) == 4:
            raw_obs_next, reward, done, info = step_out
        elif len(step_out) == 5:
            raw_obs_next, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            raise ValueError("Unexpected step output format")

        list_obs_home_next = self._split_obs(raw_obs_next)
        state_home_next = np.concatenate(list_obs_home_next, axis=0)

        state_away_next = None
        list_obs_away_next = []

        # Shared reward
        local_rewards = np.full(self.N_home, reward, dtype=np.float32)

        return (
            state_home_next,
            state_away_next,
            list_obs_home_next,
            list_obs_away_next,
            float(reward),
            local_rewards,
            done,
            info
        )

    def random_actions(self):
        acts = []
        for _ in range(self.N_home):
            a = int(self.base_env.action_space.sample())
            acts.append(a)
        return np.array(acts, dtype=int)


# Minimal Requirement (look at train_hsd.py, env_wrapper.py in hmarl, Overcooked_Env.py in ZSC-Eval)
"""
1. Configuration
env.action_dim, env.state_dim, env.obs_dim, env.N_home
2. Important Functionality of ZSC -> HSD interface
2-1. env.random_actions()
2-2. env.reset()
(
    state_home,    # global home state      (np array, shape = [state_dim])
    state_away,    # global away state      (np array OR None)
    list_obs_home, # list of agent obs      (len = N_home)
    list_obs_away, # list (empty or same as home)
    done           # False at reset
)
2-3. env.step(actions_int)
(
    state_home_next,     # np array [state_dim]
    state_away_next,     # np array or None
    list_obs_home_next,  # list of obs for each home agent
    list_obs_away_next,  # (empty or mirrored)
    reward,              # scalar float
    local_rewards,       # np array of shape [N_home]
    done,                # bool
    info                 # dict
)
3. Importation Functionality of HSD -> ZSC interface
"""

# Usage (How to replace STS2 simulator with overcooked)
"""
env = env_wrapper.Env(config_env, config_main)
---
from overcooked_env_hsd import OvercookedHSDWrapper
import overcooked_ai   # or ZSC’s Overcooked loader

base_env = <construct your Overcooked env here>
env = OvercookedHSDWrapper(base_env, num_agents=config_env['num_home_players'])
"""
