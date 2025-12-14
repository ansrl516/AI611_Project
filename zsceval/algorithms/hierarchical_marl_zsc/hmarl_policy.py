"""PyTorch implementation of hierarchical cooperative MARL with skill discovery (HSD)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import zsceval.algorithms.hierarchical_marl_zsc.utils.networks as networks


def hard_update(target, source):
    for targ_param, src_param in zip(target.parameters(), source.parameters()):
        targ_param.data.copy_(src_param.data)


# HMARL for ZSC-Eval
class HMARLModel:
    def __init__(self, cfg_m, device=torch.device("cpu")):
        """Current Implementation does not support environment batching
        Args:
            rMAPPO-style structured config:

            cfg["model"]
            cfg["learning"]
            cfg["traj_sampling"]
            cfg["network"]
            cfg["train"]

        Description of functions (functions not mentioned here are used only internally):
            Two levels of policies:
                - assign_skills: High level policy that assigns skills to agents
                - get_actions: Low level policy that selects actions given skills (requires feeding current skills)
            Trainer for both levels of policies (Q-functions) and decoder:
                - train_policy_high: Batch -> Update Q-functions for high-level policy
                - train_policy_low: Batch -> Update Q-functions for low-level policy
                - train_decoder: Dataset -> Update decoder that predicts skills from trajectories
            Helper functions:
                - compute_intrinsic_reward: Computes decoder reward
            Batched versions: add _batch suffix to above functions (only for get_actions and assign_skills because other functions allow batch input already)

        How to use: Wrap this with trainer class (e.g., HMARLWrapper) that schedules & interacts with env, ...
            - Use get_actions() and assign_skills() to run the hierarchical policy
            - Use train_policy_high(), train_policy_low(), train_decoder() to update respective networks
            - Use compute_intrinsic_reward() to get decoder-based intrinsic rewards

        """

        # Model settings
        self.num_agents = cfg_m["num_agents"]
        self.num_actions = cfg_m["num_actions"]
        self.num_skills = cfg_m["num_skills"]
        self.state_dim = cfg_m["state_dim"]
        self.obs_dim = cfg_m["obs_dim"]

        self.device = device

        # Observation shapes
        self.C = cfg_m["obs_channels"]
        self.H = cfg_m["obs_height"]
        self.W = cfg_m["obs_width"]
        self.C_share = cfg_m["share_obs_channels"]

        # Learning params
        self.gamma = cfg_m["gamma"]
        self.tau = cfg_m["tau"]
        self.lr_Q = cfg_m["lr_q"]
        self.lr_decoder = cfg_m["lr_decoder"]

        # Trajectory sampling
        self.steps_per_assign = cfg_m["steps_per_assign"]
        self.traj_skip = cfg_m["traj_skip"]
        self.use_state_difference = cfg_m["use_state_difference"]
        self.obs_truncate_length = cfg_m["obs_truncate_length"]

        # Network sizes
        self.nn = {
            "n_h_decoder": cfg_m["n_h_decoder"],
            "n_h1_low": cfg_m["n_h1_low"],
            "n_h2_low": cfg_m["n_h2_low"],
            "n_h1_high": cfg_m["n_h1_high"],
            "n_h2_high": cfg_m["n_h2_high"],
            "n_h_mixer": cfg_m["n_h_mixer"],
        }

        # ----------------------------------------------------------------------
        # Build networks
        # ----------------------------------------------------------------------
        self.obs_encoder = networks.ObsEncoder(
            self.C, self.obs_dim, self.H, self.W
        ).to(self.device)
        self.share_obs_encoder = networks.ShareObsEncoder(
            self.C_share, self.state_dim, self.H, self.W
        ).to(self.device)

        # Decoder
        self.traj_length_downsampled = int(
            np.floor(self.steps_per_assign / self.traj_skip)
        )
        decoder_input_dim = self.obs_truncate_length or self.obs_dim
        self.decoder = networks.Decoder(
            decoder_input_dim,
            self.traj_length_downsampled,
            self.nn["n_h_decoder"],
            self.num_skills,
        ).to(self.device)

        self.decoder_opt = torch.optim.Adam(
            self.decoder.parameters(), lr=self.lr_decoder
        )
        self.ce_loss = nn.CrossEntropyLoss()

        # Low-level Q-functions
        self.Q_low = networks.QLow(
            self.obs_dim,
            self.num_skills,
            self.nn["n_h1_low"],
            self.nn["n_h2_low"],
            self.num_actions,
        ).to(self.device)
        self.Q_low_target = networks.QLow(
            self.obs_dim,
            self.num_skills,
            self.nn["n_h1_low"],
            self.nn["n_h2_low"],
            self.num_actions,
        ).to(self.device)
        hard_update(self.Q_low_target, self.Q_low)
        self.low_opt = torch.optim.Adam(self.Q_low.parameters(), lr=self.lr_Q)

        # High-level Qmix (agent utilities from local obs, mixer from shared obs)
        self.agent_main = networks.QmixSingle(
            self.obs_dim,
            self.nn["n_h1_high"],
            self.nn["n_h2_high"],
            self.num_skills,
        ).to(self.device)
        self.agent_target = networks.QmixSingle(
            self.obs_dim,
            self.nn["n_h1_high"],
            self.nn["n_h2_high"],
            self.num_skills,
        ).to(self.device)

        self.mixer_main = networks.QmixMixer(
            self.state_dim, self.num_agents, self.nn["n_h_mixer"]
        ).to(self.device)
        self.mixer_target = networks.QmixMixer(
            self.state_dim, self.num_agents, self.nn["n_h_mixer"]
        ).to(self.device)
        hard_update(self.agent_target, self.agent_main)
        hard_update(self.mixer_target, self.mixer_main)

        self.high_opt = torch.optim.Adam(
            list(self.agent_main.parameters()) + list(self.mixer_main.parameters()),
            lr=self.lr_Q,
        )
        self.loss_fn = nn.MSELoss()

        # Internal states
        self.current_skills = None
        self.step = 0

    ## --- API functions for using it as pretrained policy pool inside separated overcooked runner --- ##

    @torch.no_grad()
    def get_actions(
        self,
        share_obs,
        obs,
        rnn_states,
        rnn_states_critic,
        masks,
        available_actions,
    ):
        # Dummies: rnn_states, rnn_states_critic, action_log_prob, masks
        """
        Inputs:
            share_obs: (batch, num_agents, H, W, C_share)
            obs:       (batch, num_agents, H, W, C)
            rnn_states: (batch, num_agents, rnn_N, hidden)  (unused placeholder)
            rnn_states_critic: (batch, num_agents, rnn_N, hidden)  (unused placeholder)
            masks:      (batch, 1)
            available_actions: (batch, num_agents, num_actions)

        Outputs (all tensors):
            value:              (batch, num_agents, 1)  (dummy zeros)
            action:             (batch, num_agents, 1)  (discrete index per agent)
            action_log_prob:    (batch, num_agents, 1)  (dummy zeros)
            next_rnn_state:     (batch, num_agents, rnn_N, hidden) (zeros)
            next_rnn_state_cr:  (batch, num_agents, rnn_N, hidden) (zeros)
        """

        batch_size = obs.shape[0]
        device = obs.device

        # value (dummy critic)
        value = torch.zeros((batch_size, self.num_agents, 1), device=device)

        # actions from hierarchical policy
        action = self.get_actions_algorithm(
            steps=self.step,
            obs=obs,
            shared_obs=share_obs,
            available_actions=available_actions,
            epsilon=0.0,
        )

        # dummy log-probs and rnn states
        action_log_prob = torch.zeros((batch_size, self.num_agents, 1), device=device)
        next_rnn_state = torch.zeros_like(rnn_states)
        next_rnn_state_critic = torch.zeros_like(rnn_states_critic)

        self.step += 1

        return value, action, action_log_prob, next_rnn_state, next_rnn_state_critic

    # getting fixed actions for this policy (only used as fixed)
    def act(self, obs, rnn_state, mask, available_actions=None, deterministic=True):
        # Dummies: rnn_state, mask, deterministic
        action = self.get_actions_algorithm(
            steps=self.step,
            obs=obs,
            shared_obs=obs,
            available_actions=available_actions,
            epsilon=0.0,
        )
        next_rnn_state = rnn_state
        self.step += 1
        return action, next_rnn_state

    # dummy function for API compatibility
    def lr_decay(self, episode, total):
        pass  # no-op

    ## --- End of API functions --- ##

    ## --- Core action functions for hierarchical MARL with skill discovery --- ##

    @torch.no_grad()
    def get_actions_algorithm(self, steps, obs, shared_obs, available_actions, epsilon=None):
        """Wraps get_actions_low and assign_skills with internal variables, implements HMARL logic."""
        # 1. Assign skills at the beginning and every steps_per_assign
        if steps % self.steps_per_assign == 0:
            self.current_skills = self.assign_skills(obs=obs, share_obs=shared_obs, epsilon=epsilon)  # [B, N]

        # 2. Get low-level actions using current skills
        actions = self.get_actions_low(obs, available_actions, self.current_skills, epsilon=epsilon)  # [B, N]

        return self._format_actions(actions)  # [B, N, 1]

    def get_actions_low(self, obs, available_actions, skills, epsilon=None):
        """
        Compute low-level control actions conditioned on current skills.

        Args:
            obs:               (B, N, H, W, C)
            available_actions: (B, N, A)
            skills:            (B, N) int skill indices
            epsilon:           optional exploration rate for low-level (float)

        Returns:
            actions: (B, N) int actions
        """

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)          # (B, N, H, W, C)
        skills_t = torch.as_tensor(skills, dtype=torch.long, device=self.device)       # (B, N)
        avail_t = torch.as_tensor(
            available_actions, dtype=torch.bool, device=self.device
        )  # (B, N, A)

        # encode observations
        obs_encoded = self.obs_encoder(obs_t)  # (B, N, obs_dim)

        # one-hot skills
        skills_onehot = F.one_hot(skills_t, num_classes=self.num_skills).float()  # (B, N, K)

        if epsilon is None:
            epsilon = 0.0

        with torch.no_grad():
            q = self.Q_low(obs_encoded, skills_onehot)  # (B, N, A)

            # mask unavailable actions
            q_masked = q.masked_fill(~avail_t, float("-inf"))

            # greedy per-agent action
            actions = torch.argmax(q_masked, dim=2)  # (B, N)

            actions_np = actions.cpu().numpy()

            # epsilon-greedy exploration over available actions
            if epsilon > 0.0:
                avail_np = avail_t.cpu().numpy()
                explore_mask = np.random.rand(*actions_np.shape) < epsilon
                explore_indices = np.argwhere(explore_mask)
                for b, n in explore_indices:
                    avail = avail_np[b, n]
                    if avail.any():
                        actions_np[b, n] = np.random.choice(np.nonzero(avail)[0])
                    else:
                        actions_np[b, n] = np.random.randint(self.num_actions)

        return actions_np

    def assign_skills(self, obs, share_obs=None, epsilon=None):
        """
        Assign skills to agents via high-level Q-network.
        Args:
            obs: np array of shape [B, N, H, W, C]
            share_obs: optional np array of shape [B, N, H, W, C_share]
            epsilon: exploration rate (float or None)
        Returns:
            skills: np array of shape [B, N]
        """

        B = obs.shape[0]
        N = self.num_agents

        if epsilon is None:
            epsilon = 0.0

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)  # (B, N, H, W, C)

        # Per-agent utilities use local observation encoding (matches training).
        obs_encoded = self.obs_encoder(obs_t)  # (B, N, obs_dim)

        with torch.no_grad():
            q_values = self.agent_main(obs_encoded)  # (B, N, K)
            greedy_skills = torch.argmax(q_values, dim=2).cpu().numpy()  # (B, N)

        # epsilon-greedy per agent
        explore_mask = np.random.rand(B, N) < epsilon
        random_skills = np.random.randint(0, self.num_skills, size=(B, N))
        skills = np.where(explore_mask, random_skills, greedy_skills)

        return skills.astype(np.int32)

    ## --- End of Core action functions --- ##

    ## --- Training related functions for hierarchical MARL with skill discovery --- ##

    def process_batch_high(self, batch):
        """
        batch: list of transitions, each:
            [obs_h, share_obs_h, skills, reward_h, obs_next_h, share_obs_next_h, done_env]

        Shapes: 
            obs_h:              (B_env, N, H, W, C)
            share_obs_h:        (B_env, N, H, W, C_share)
            skills:             (B_env, N)
            reward_h:           (B_env,)   scalar reward per environment
            obs_next_h:         (B_env, N, H, W, C)
            share_obs_next_h:   (B_env, N, H, W, C_share)
            done_env:           (B_env,)
        """

        # Unzip
        obs_list, share_list, skills_list, reward_list, obsn_list, sharen_list, done_list = zip(*batch)

        # Stack
        obs = np.stack(obs_list, axis=0)            # (B_s, B_env, N, H, W, C)
        share_obs = np.stack(share_list, axis=0)    # (B_s, B_env, N, H, W, C_share)
        skills = np.stack(skills_list, axis=0)      # (B_s, B_env, N)
        rewards = np.stack(reward_list, axis=0)     # (B_s, B_env)
        obs_next = np.stack(obsn_list, axis=0)      # (B_s, B_env, N, H, W, C)
        share_obs_next = np.stack(sharen_list, axis=0)
        dones = np.stack(done_list, axis=0)         # (B_s, B_env)

        B_s, B_env, N, H, W, C = obs.shape
        B_total = B_s * B_env

        # Flatten sample × env dims
        obs = obs.reshape(B_total, N, H, W, C)
        share_obs = share_obs.reshape(B_total, N, H, W, share_obs.shape[-1])
        skills = skills.reshape(B_total, N)
        rewards = rewards.reshape(B_total)
        obs_next = obs_next.reshape(B_total, N, H, W, C)
        share_obs_next = share_obs_next.reshape(B_total, N, H, W, share_obs_next.shape[-1])
        dones = dones.reshape(B_total)

        # Convert to torch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        share_obs_t = torch.as_tensor(share_obs, dtype=torch.float32, device=self.device)
        skills_t = torch.as_tensor(skills, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        share_obs_next_t = torch.as_tensor(share_obs_next, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Encode
        obs_enc = self.obs_encoder(obs_t)                      # (B_total, N, obs_dim)
        share_enc = self.share_obs_encoder(share_obs_t)        # (B_total, N, state_dim)
        obs_next_enc = self.obs_encoder(obs_next_t)
        share_next_enc = self.share_obs_encoder(share_obs_next_t)

        # QMIX mixer expects a single state vector per sample (not per-agent),
        # so collapse the agent dimension of the shared encodings.
        state_enc = share_enc.mean(dim=1)              # (B_total, state_dim)
        state_next_enc = share_next_enc.mean(dim=1)    # (B_total, state_dim)

        # One-hot skills
        skills_oh = F.one_hot(skills_t, num_classes=self.num_skills).float()

        return (
            B_total,
            state_enc,        # state encoding for mixer
            obs_enc,          # obs encoding for agent utility
            skills_oh,
            rewards_t,
            state_next_enc,
            obs_next_enc,
            dones_t,
        )

    def train_policy_high(self, batch):
        # batch: np array of [obs_h, share_obs_h, current_skills, rewards_h, next_obs, next_share_obs, done_env]
        (
            n_samples,
            state,
            obs,
            skills_1hot,
            reward,
            state_next,
            obs_next,
            done,
        ) = self.process_batch_high(batch)

        # one-step TD for joint Q_tot
        with torch.no_grad():
            # target per-agent utilities for next obs
            agent_q_next = self.agent_target(obs_next)  # (B, N, K)

            # greedy skills for target
            argmax_actions = torch.argmax(agent_q_next, dim=2)  # (B, N)
            skills_target_1hot = F.one_hot(
                argmax_actions, num_classes=self.num_skills
            ).float()  # (B, N, K)

            # Q values for chosen target skills
            q_target_selected = (agent_q_next * skills_target_1hot).sum(dim=2)  # (B, N)

            # Mixer expects (B, N) for agent Qs and state encoding
            mixer_input_target = q_target_selected.view(-1, self.num_agents)  # (B, N)
            q_tot_target = self.mixer_target(mixer_input_target, state_next).squeeze(1)  # (B,)

            # done multiplier: 1 for non-terminal, 0 for terminal
            done_multiplier = 1.0 - done  # (B,)

            target = reward + self.gamma * q_tot_target * done_multiplier  # (B,)

        # current Q_tot
        agent_q = self.agent_main(obs)  # (B, N, K)
        q_selected = (agent_q * skills_1hot).sum(dim=2)  # (B, N)
        mixer_input = q_selected.view(-1, self.num_agents)  # (B, N)
        q_tot = self.mixer_main(mixer_input, state).squeeze(1)  # (B,)

        loss = self.loss_fn(q_tot, target)
        self.high_opt.zero_grad()
        loss.backward()
        self.high_opt.step()

        networks.soft_update(self.agent_target, self.agent_main, self.tau)
        networks.soft_update(self.mixer_target, self.mixer_main, self.tau)

    def process_batch_low(self, batch):
        """
        batch: list of transitions, each transition = 
            [obs, actions, rewards_low, skills, next_obs, done_env]

        Shapes per transition element:
            obs:        (B_env, N, H, W, C)
            actions:    (B_env, N)
            rewards:    (B_env, N)
            skills:     (B_env, N)
            next_obs:   (B_env, N, H, W, C)
            done_env:   (B_env,) or (B_env, N) or (B_env, N, 1)

        We convert to:
            obs_enc:        (B_total, N, obs_dim)
            actions_1hot:   (B_total, N, A)
            rewards_t:      (B_total, N)
            obs_next_enc:   (B_total, N, obs_dim)
            skills_oh:      (B_total, N, K)
            dones_t:        (B_total,)
        where:
            B_total = len(batch) * B_env
        """

        # --- 1. Unzip transitions ---
        obs_list, act_list, rew_list, skill_list, next_obs_list, done_list = zip(*batch)

        # --- 2. Stack each field ---
        obs = np.stack(obs_list, axis=0)            # (B_s, B_env, N, H, W, C)
        actions = np.stack(act_list, axis=0)        # (B_s, B_env, N)
        rewards = np.stack(rew_list, axis=0)        # (B_s, B_env, N)
        skills = np.stack(skill_list, axis=0)       # (B_s, B_env, N)
        obs_next = np.stack(next_obs_list, axis=0)  # (B_s, B_env, N, H, W, C)
        dones = np.stack(done_list, axis=0)         # (B_s, B_env, [N]...)

        B_s, B_env, N, H, W, C = obs.shape
        B_total = B_s * B_env

        # --- 3. Flatten sample & env dims ---
        obs = obs.reshape(B_total, N, H, W, C)
        obs_next = obs_next.reshape(B_total, N, H, W, C)
        actions = actions.reshape(B_total, N)
        rewards = rewards.reshape(B_total, N)
        skills = skills.reshape(B_total, N)

        # --- 4. Normalize dones to (B_total,) ---
        dones_arr = np.asarray(dones)
        # Case: (B_s, B_env)
        if dones_arr.ndim == 2:  # (B_s, B_env)
            dones_flat = dones_arr.reshape(B_total)
        else:
            # Cases: (B_s, B_env, N) or (B_s, B_env, N, 1)
            dones_flat = np.any(dones_arr.reshape(B_s, B_env, -1) > 0.5, axis=2)
            dones_flat = dones_flat.reshape(B_total)

        # --- 5. Convert to torch ---
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        skills_t = torch.as_tensor(skills, dtype=torch.int64, device=self.device)
        dones_t = torch.as_tensor(dones_flat, dtype=torch.float32, device=self.device)

        # --- 6. Encode obs ---
        obs_enc = self.obs_encoder(obs_t)           # (B_total, N, obs_dim)
        obs_next_enc = self.obs_encoder(obs_next_t)  # (B_total, N, obs_dim)

        # --- 7. One-hot encodings ---
        skills_oh = F.one_hot(skills_t, num_classes=self.num_skills).float()
        actions_1hot = F.one_hot(actions_t, num_classes=self.num_actions).float()

        return (
            B_total,
            obs_enc,
            actions_1hot,
            rewards_t,
            obs_next_enc,
            skills_oh,
            dones_t,
        )

    def train_policy_low(self, batch):
        # batch: np array of [obs, actions_int, rewards_low, skills_int, obs_next, done_env]
        (
            n_samples,
            obs,
            actions_1hot,
            rewards,
            obs_next,
            skills,
            done,
        ) = self.process_batch_low(batch)

        B = obs.shape[0]

        # one-step TD for low-level Q
        with torch.no_grad():
            # Double Q: online net selects, target net evaluates
            q_next_online = self.Q_low(obs_next, skills)  # (B, N, A)
            next_actions = torch.argmax(q_next_online, dim=2, keepdim=True)  # (B, N, 1)

            q_target_all = self.Q_low_target(obs_next, skills)  # (B, N, A)
            q_target_selected = torch.gather(
                q_target_all, 2, next_actions
            ).squeeze(2)  # (B, N)

            # done is per-env; expand to per-agent
            done_expanded = done.view(B, 1)  # (B, 1)
            done_expanded = done_expanded.expand_as(q_target_selected)  # (B, N)
            done_mult = 1.0 - done_expanded

            target = rewards + self.gamma * q_target_selected * done_mult  # (B, N)

        q_all = self.Q_low(obs, skills)  # (B, N, A)
        q_selected = (q_all * actions_1hot).sum(dim=2)  # (B, N)

        loss = self.loss_fn(q_selected, target)
        self.low_opt.zero_grad()
        loss.backward()
        self.low_opt.step()

        networks.soft_update(self.Q_low_target, self.Q_low, self.tau)

    def _downsample_traj(self, obs):
        """Helper for decoder (obs shape: [B, T, obs_dim])."""
        obs_downsampled = obs[:, :: self.traj_skip, :]
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[:, :, : self.obs_truncate_length]
        if self.use_state_difference:
            obs_downsampled = obs_downsampled[:, 1:, :] - obs_downsampled[:, :-1, :]
        assert (
            obs_downsampled.shape[1] == self.traj_length_downsampled
        ), f"Downsampled traj length {obs_downsampled.shape[1]} != expected {self.traj_length_downsampled}"
        return obs_downsampled

    def train_decoder(self, dataset):
        """
        dataset: list of [traj_slice, skill_id]
            traj_slice: (T_i, H, W, C)
            skill_id: scalar int
        """

        traj_list = [item[0] for item in dataset]
        skill_list = [item[1] for item in dataset]

        B = len(traj_list)
        if B == 0:
            return 0.0

        T_target = self.steps_per_assign
        _, H, W, C = traj_list[0].shape

        padded_traj = np.zeros((B, T_target, H, W, C), dtype=np.float32)
        for i, traj in enumerate(traj_list):
            T_i = traj.shape[0]
            if T_i >= T_target:
                padded_traj[i] = traj[:T_target]
            else:
                padded_traj[i, :T_i] = traj

        traj_np = padded_traj  # (B, T_target, H, W, C)
        skill_np = np.asarray(skill_list, dtype=np.int64)

        traj_t = torch.as_tensor(traj_np, dtype=torch.float32, device=self.device)
        skills_t = torch.as_tensor(skill_np, dtype=torch.long, device=self.device)

        B, T, _, _, _ = traj_np.shape
        traj_flat = traj_t.reshape(B * T, H, W, C)

        with torch.no_grad():
            obs_flat = self.obs_encoder(traj_flat)  # (B*T, obs_dim)

        obs_seq = obs_flat.reshape(B, T, -1)  # (B, T, obs_dim)
        traj_down = self._downsample_traj(obs_seq)

        logits, probs = self.decoder(traj_down)
        loss = self.ce_loss(logits, skills_t)

        self.decoder_opt.zero_grad()
        loss.backward()
        self.decoder_opt.step()

        with torch.no_grad():
            expected_prob = probs.gather(1, skills_t.unsqueeze(1)).mean().item()

        return expected_prob

    def use_decoder(obs):
        """
        Compute decoder output given trajectory observations.
        input:
            [batch_size(=rollout_threads), num_agents, period of high policy, H, W, C]
        output:
            [batch_size, num_agents, num_skills]

        """
        pass

    def compute_intrinsic_reward(self, agents_traj_obs, skills):
        """
        Compute decoder-based intrinsic reward:
        input:
            agents_traj_obs: np.ndarray of shape (B, T, H, W, C)
            skills: shape (B,), skill index for each agent-trajectory

        output:
            reward: (B,) numpy array
        """

        traj_np = np.asarray(agents_traj_obs)
        if traj_np.ndim != 5:
            raise ValueError(
                f"[compute_intrinsic_reward] Expected 5D input (B,T,H,W,C), got shape {traj_np.shape}"
            )

        B, T, H, W, C = traj_np.shape

        traj_flat = traj_np.reshape(B * T, H, W, C)
        traj_flat_t = torch.as_tensor(
            traj_flat, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            obs_encoded = self.obs_encoder(traj_flat_t)  # (B*T, obs_dim)

        obs_encoded = obs_encoded.reshape(B, T, -1)  # (B, T, obs_dim)

        traj_down = self._downsample_traj(obs_encoded)

        skills_t = torch.as_tensor(skills, dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, decoder_probs = self.decoder(traj_down)  # (B, num_skills)
            prob = decoder_probs[torch.arange(B), skills_t]  # (B,)

        return prob.cpu().numpy()
        # return np.zeros_like(prob.cpu().numpy()) # return zero

    @torch.no_grad()
    def reset(self):
        """Reset internal step counter, called at beginning of each episode or eval."""
        self.step = 0

    @staticmethod
    def _format_actions(actions: np.ndarray) -> np.ndarray:
        """
        Ensure actions always have shape (..., num_agents, 1) so env's _action_convertor
        receives indexable entries. So we add a dummy last dimension.
        """
        return np.expand_dims(actions, axis=-1)

    def save(self, path):
        print("saving to", path)
        torch.save(
            {
                "decoder": self.decoder.state_dict(),
                "Q_low": self.Q_low.state_dict(),
                "Q_low_target": self.Q_low_target.state_dict(),
                "agent_main": self.agent_main.state_dict(),
                "agent_target": self.agent_target.state_dict(),
                "mixer_main": self.mixer_main.state_dict(),
                "mixer_target": self.mixer_target.state_dict(),
                "obs_encoder": self.obs_encoder.state_dict(),
                "share_obs_encoder": self.share_obs_encoder.state_dict(),
            },
            path,
        )

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.Q_low.load_state_dict(checkpoint["Q_low"])
        self.Q_low_target.load_state_dict(checkpoint["Q_low_target"])
        self.agent_main.load_state_dict(checkpoint["agent_main"])
        self.agent_target.load_state_dict(checkpoint["agent_target"])
        self.mixer_main.load_state_dict(checkpoint["mixer_main"])
        self.mixer_target.load_state_dict(checkpoint["mixer_target"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.share_obs_encoder.load_state_dict(checkpoint["share_obs_encoder"])

    def prep_rollout(self):
        self.decoder.eval()
        self.Q_low.eval()
        self.agent_main.eval()
        self.mixer_main.eval()
        self.obs_encoder.eval()
        self.share_obs_encoder.eval()

        # reset internal skill storage
        self.reset()

    def prep_training(self):
        self.decoder.train()
        self.Q_low.train()
        self.agent_main.train()
        self.mixer_main.train()
        self.obs_encoder.train()
        self.share_obs_encoder.train()
