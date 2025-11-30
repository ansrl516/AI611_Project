"""PyTorch implementation of hierarchical cooperative MARL with skill discovery (HSD)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils.networks as networks


def hard_update(target, source):
    for targ_param, src_param in zip(target.parameters(), source.parameters()):
        targ_param.data.copy_(src_param.data)
 
# HMARL for ZSC-Eval which includes update methods
# Add encoder of observations (TODO) to match the shape
class HMARLModel: 
    def __init__(self, param_sharing_option, config_constants, config_traj_sampling, num_agents, 
                 state_dim, obs_dim, num_actions, num_skills, config_nn, device=None):
        """Current Implmentation does not support environment batching
        Args:
            param_sharing_option: parameter sharing option for decentralized training (for low-level policy, high-level policy)
            config_constants: dictionary of {learning rate, update rate, discounting factor in RL}
            config_traj_sampling: dictionary of trajectory downsampling options for decoder
            num_agents: number of agents on the team controlled by this alg
            state_dim, obs_dim, num_actions, num_skills: shared_obs dim, policy_obs dim, number of action, 
            config_nn: dictionary with neural net sizes

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
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.num_actions = num_actions # actions are one-hot encoded

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configuration
        self.nn = config_nn
        self.num_skills = num_skills # dimension of the high skill variable (one-hot encoded)
        self.tau = config_constants["tau"]
        self.lr_Q = config_constants["lr_Q"]
        self.lr_decoder = config_constants["lr_decoder"]
        self.gamma = config_constants["gamma"]

        # Model for encoding obs 
        # "all_agent_obs" : np.array of shape (num_agents, H, W, C) -> (num_agents, obs_dim)
        # "share_obs" : np.array of shape (num_agents, H, W, C_share) -> (num_agents, state_dim)
        self.obs_encoder = networks.ObsEncoder(self.C, self.obs_dim, self.H, self.W).to(self.device)
        self.share_obs_encoder = networks.ShareObsEncoder(self.C_share, self.state_dim, self.H, self.W).to(self.device)

        # Decoder settings
        self.traj_length = config_traj_sampling["steps_per_assign"]
        self.traj_skip = config_traj_sampling["traj_skip"]
        self.traj_length_downsampled = int(np.ceil(self.traj_length / self.traj_skip))
        self.use_state_difference = config_traj_sampling["use_state_difference"]
        if self.use_state_difference:
            self.traj_length_downsampled -= 1
        self.obs_truncate_length = config_traj_sampling["obs_truncate_length"]
        assert (self.obs_truncate_length is None) or (self.obs_truncate_length <= self.obs_dim)

        # Decoder Network
        decoder_input_dim = self.obs_truncate_length or self.obs_dim
        self.decoder = networks.Decoder(decoder_input_dim, self.traj_length_downsampled, self.nn["n_h_decoder"], self.num_skills).to(self.device)
        
        # Decoder Optimizer and Loss
        self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=self.lr_decoder)
        self.ce_loss = nn.CrossEntropyLoss()

        # Currently, only parameter sharing option supported is "all_shared"
        assert param_sharing_option == "all_shared", "Only 'all_shared' parameter sharing option is supported currently."

        # Low-level Q-functions (target stands for moving average update)
        self.Q_low = networks.QLow(self.obs_dim, self.num_skills, self.nn["n_h1_low"], self.nn["n_h2_low"], self.num_actions).to(self.device)
        self.Q_low_target = networks.QLow(self.obs_dim, self.num_skills, self.nn["n_h1_low"], self.nn["n_h2_low"], self.num_actions).to(self.device)

        # High-level Q-functions & Q-Centralized Mixer
        self.agent_main = networks.QmixSingle(self.obs_dim, self.nn["n_h1"], self.nn["n_h2"], self.num_skills).to(self.device)
        self.agent_target = networks.QmixSingle(self.obs_dim, self.nn["n_h1"], self.nn["n_h2"], self.num_skills).to(self.device)
        self.mixer_main = networks.QmixMixer(self.state_dim, self.num_agents, self.nn["n_h_mixer"]).to(self.device)
        self.mixer_target = networks.QmixMixer(self.state_dim, self.num_agents, self.nn["n_h_mixer"]).to(self.device)
        
        # Optimizer and Loss for Q-functions
        hard_update(self.Q_low_target, self.Q_low)
        self.low_opt = optim.Adam(self.Q_low.parameters(), lr=self.lr_Q)

        hard_update(self.agent_target, self.agent_main)
        hard_update(self.mixer_target, self.mixer_main)
        self.high_opt = optim.Adam(list(self.agent_main.parameters()) + list(self.mixer_main.parameters()), lr=self.lr_Q)
        self.loss_fn = nn.MSELoss()

        # Maintain internal storage of current skills
        self.current_skills = None
    
    ## --- API functions for using it as pretrained policy pool inside separated overcooked runner --- ##
    
    # getting fixed actions for this policy (only used as fixed)
    @torch.no_grad()
    def get_actions(self, share_obs, obs, rnn_states, rnn_states_critic, masks, available_actions):
        # Dummies: rnn_states, rnn_states_critic, action_log_prob, masks
        """
        Inputs:
            share_obs: (n_rollouts, obs_dim)
            obs:       (n_rollouts, obs_dim)  ← used by actor
            rnn_states: (n_rollouts, rnn_N, hidden)
            rnn_states_critic: (n_rollouts, rnn_N, hidden)
            masks:      (n_rollouts, 1)
            available_actions: (n_rollouts, num_actions)
        
        Outputs (all tensors):
            value:              (n_rollouts, 1)
            action:             (n_rollouts, act_dim)
            action_log_prob:    (n_rollouts, act_dim) or (n_rollouts, 1)
            next_rnn_state:     (n_rollouts, rnn_N, hidden)
            next_rnn_state_cr:  (n_rollouts, rnn_N, hidden)
        """

        n_rollouts = obs.shape[0]
        device = obs.device

        # ---- VALUE (dummy, fixed policy has no critic) ----
        # MUST be shape (n_rollouts, 1)
        value = torch.zeros((n_rollouts, 1), device=device)

        # ---- ACTION ----
        # Fixed pretrained policy uses a feedforward actor
        # actor(obs) must return shape: (n_rollouts, act_dim)
        action = self.get_actions_algorithm(obs, available_actions, epsilon=0.0)    # Already correct shape

        # ---- ACTION LOG PROB ----
        # A fixed deterministic policy can return zero logs
        action_log_prob = torch.zeros((n_rollouts, 1), device=device)

        # ---- NEXT RNN STATES ----
        # Must match shape of input state
        next_rnn_state = torch.zeros_like(rnn_states)           # safe dummy
        next_rnn_state_critic = torch.zeros_like(rnn_states_critic)

        return value, action, action_log_prob, next_rnn_state, next_rnn_state_critic

    # getting fixed actions for this policy (only used as fixed)
    def act(self, obs, rnn_state, mask, available_actions=None, deterministic=True):
        # Dummies: rnn_state, mask, deterministic
        action = self.get_actions_algorithm(obs, available_actions, epsilon=0.0)
        next_rnn_state = rnn_state
        return action, next_rnn_state

    # dummy function for API compatibility
    def lr_decay(self, episode, total):
        pass   # no-op    
    
    ## --- End of API functions --- ##

    ## --- Core action functions for hierarchical MARL with skill discovery --- ##

    @torch.no_grad()
    def get_actions_algorithm(self, steps, obs, shared_obs, available_actions, epsilon=None):
        """ Wraps get_actions_low and assign_skills with internal variables, implements hmarl logic. """
        # 1. Assign skills at the beginning and every steps_per_assign
        if steps % self.steps_per_assign == 0:
            self.current_skills = self.assign_skills(shared_obs, epsilon=epsilon)

        # 2. Get low-level actions using current skills
        actions = self.get_actions_low(obs, available_actions, self.current_skills)

        return actions

    def get_actions_low(self, list_obs, available_actions, skills):
        """Get low-level actions for all agents as a batch. Used in get_actions_algorithm. """
        # Shape of list_obs: [n_rollouts, num_agents, H, W, C] where each entry stands for individual observation of that agent
        # Shape of skills: [n_rollouts, num_agents,] where each entry is int skill of that agent
        # Shape of available_actions: [n_rollouts, num_agents, num_actions] where each entry is binary mask of available actions
        # Shape of actions: [n_rollouts, num_agents] where each entry is int action of that agent 

        # Transform into tensors & input form of obs_encoder, Q_low 
        obs = torch.as_tensor(np.array(list_obs), dtype=torch.float32, device=self.device) # shape [n_rollouts, num_agents, obs_dim] (obs per agent)
        skills_t = torch.as_tensor(np.array(skills), dtype=torch.float32, device=self.device) # shape [n_rollouts, num_agents] where each entry is int skill of that agent
        skills_t = torch.nn.functional.one_hot(skills_t.long(), num_classes=self.num_skills).float() # shape [n_rollouts, num_agents, num_skills]
        available_actions_t = torch.as_tensor(np.array(available_actions), dtype=torch.bool, device=self.device) # shape [n_rollouts, num_agents, num_actions]

        # encoder list_obs: [n_rollouts, num_agents, H, W, C] -> [n_rollouts, num_agents, obs_dim]
        list_obs = self.obs_encoder(list_obs, device=self.device)

        # get Q-values for each low level action given skill
        with torch.no_grad():
            # API requirement of Q_low: obs: [B, N, obs_dim], skills: [B, N, num_skills] -> Q-values: [B, N, num_actions]
            q_values = self.Q_low(obs, skills_t) 
            masked_q = q_values.masked_fill(~available_actions_t, float('-inf'))
            greedy_actions = torch.argmax(masked_q, dim=2).cpu().numpy() # select greedy low-level action per agent within available actions

        actions = np.zeros((self.batch, self.num_agents), dtype=int)
        for idx in range(self.num_agents):
            actions[:, idx] = greedy_actions[:, idx]
        return actions

    def assign_skills(self, share_obs, epsilon=None): 
        """ Assign skills to agents using high-level policy. Used in get_actions_algorithm. """
        # share_obs: [n_rollouts, num_agents, H, W , C_share] where this is the shared observation available to all agents
        # skills: [n_rollouts, num_agents,] where each entry is int skill of that agent

        # Transform into tensors & input form of share_obs_encoder
        share_obs = torch.as_tensor(np.array(share_obs), dtype=torch.float32, device=self.device) # shape [n_rollouts, num_agents, obs_dim] (obs per agent)
        
        # API requirement of share_obs_encoder: share_obs: [B, N, H, W, C_share] -> [B, N, state_dim]
        share_obs = self.share_obs_encoder(share_obs, device=self.device)

        with torch.no_grad():
            q_values = self.agent_main(share_obs)[:, :self.num_skills] # get Q-values for each skill
            skills_argmax = torch.argmax(q_values, dim=2).cpu().numpy() # select greedy skill per agent

        # ε-greedy exploration
        skills = np.zeros((self.batch, self.num_agents), dtype=int)
        for b in range(self.batch):
            for i in range(self.num_agents):
                if np.random.rand() < epsilon:
                    skills[b, i] = np.random.randint(0, self.num_skills)
                else:
                    skills[b, i] = int(skills_argmax[b, i])

        return skills

    ## --- End of Core action functions --- ##

    ## --- Training related functions for hierarchical MARL with skill discovery --- ##

    def process_batch_high(self, batch): # helper function for high-level policy training
        # batch: n_steps of [obs_h, share_obs_h, current_skills, rewards_h, next_obs, next_share_obs, done]
        # shape of obs_h: [batch, num_agents, H, W, C]
        # shape of share_obs_h: [batch, num_agents, H, W, C_share]
        # shape of current_skills: [batch, num_agents,] (int skill per agent)
        # shape of rewards_h: [batch,]
        # shape of next_obs: [batch, num_agents, H, W, C]
        # shape of next_share_obs: [batch, num_agents, H, W, C_share]
        # shape of done: [batch,] (episode termination flag 1 0)

        assert batch.shape[1] == 8, "Batch shape incorrect for high-level policy training."

        # merge n_steps and batch dimensions
        batch = np.asarray(batch) # shape [n_steps, batch, 8] where each i in [8] holds python obj
        batch = batch.reshape(-1, batch.shape[2]) # shape [n_steps * batch, ...]

        # organize batch data
        obs = np.stack(batch[:, 0]) # shape [n_steps * batch, num_agents, H, W, C]
        state = np.stack(batch[:, 1]) # shape [n_steps * batch, num_agents, H, W, C_share]
        skills = np.stack(batch[:, 2]) # shape [n_steps * batch, num_agents,]
        reward = np.stack(batch[:, 3]) # shape [n_steps * batch,]
        obs_next = np.stack(batch[:, 4]) # shape [n_steps * batch, num_agents, H, W, C]
        state_next = np.stack(batch[:, 5]) # shape [n_steps * batch, num_agents, H, W, C_share]
        done = np.stack(batch[:, 6])

        # change them into tensors
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        skills_t = torch.as_tensor(skills, dtype=torch.int64, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        state_next_t = torch.as_tensor(state_next, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # encode obs and state using encoder
        obs_t = self.obs_encoder(obs_t, device=self.device) # change obs_t into shape [n_steps * batch, num_agents, obs_dim]
        state_t = self.share_obs_encoder(state_t, device=self.device) # change state_t into shape [n_steps * batch, num_agents, state_dim]
        obs_next_t = self.obs_encoder(obs_next_t, device=self.device) # change obs_next_t into shape [n_steps * batch, num_agents, obs_dim]
        state_next_t = self.share_obs_encoder(state_next_t, device=self.device) # change state_next_t into shape [n_steps * batch, num_agents, state_dim]

        # change skills_t into one-hot (num_steps * batch, num_agents, num_skills)
        skills_t = torch.nn.functional.one_hot(skills_t, num_classes=self.num_skills).float()

        return batch.shape[0], state_t, obs_t, skills_t, reward_t, state_next_t, obs_next_t, done_t

    def train_policy_high(self, batch):
        # batch shape: npy list of [batch, share_obs_high, policy_obs_high, skills_int, rewards_high, share_obs, policy_obs, done]
        
        # process batch data 
        n_steps, state, obs, skills_1hot, reward, state_next, obs_next, done = self.process_batch_high(batch)

        with torch.no_grad(): # one step TD
            argmax_actions = torch.argmax(self.agent_target(obs_next), dim=1)
            skills_target_1hot = torch.zeros(
                (n_steps * self.num_agents, self.num_skills), dtype=torch.float32, device=self.device
            )
            skills_target_1hot.scatter_(1, argmax_actions.unsqueeze(1), 1.0)

            q_target_selected = (self.agent_target(obs_next) * skills_target_1hot).sum(dim=1)
            mixer_input_target = q_target_selected.view(-1, self.num_agents)

            state_next_t = torch.as_tensor(state_next, dtype=torch.float32, device=self.device)
            q_tot_target = self.mixer_target(mixer_input_target, state_next_t).squeeze(1)

            done_multiplier = -(torch.as_tensor(done, dtype=torch.float32, device=self.device) - 1.0)
            target = torch.as_tensor(reward, dtype=torch.float32, device=self.device) + self.gamma * q_tot_target * done_multiplier

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        skills_1hot_t = torch.as_tensor(skills_1hot, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        q_selected = (self.agent_main(obs_t) * skills_1hot_t).sum(dim=1)
        mixer_input = q_selected.view(-1, self.num_agents)
        q_tot = self.mixer_main(mixer_input, state_t).squeeze(1)

        loss = self.loss_fn(q_tot, target)
        self.high_opt.zero_grad()
        loss.backward()
        self.high_opt.step()

        networks.soft_update(self.agent_target, self.agent_main, self.tau)
        networks.soft_update(self.mixer_target, self.mixer_main, self.tau)

    def process_batch_low(self, batch): # helper function for low-level policy training
        # batch content: n_steps of [obs, actions, rewards_low, current_skills, next_obs, dones]
        # shape of obs: [batch, num_agents, H, W, C]
        # shape of actions: [batch, num_agents] (int action per agent)
        # shape of rewards_low: [batch, num_agents]
        # shape of skills_int: [batch, num_agents] (int skill per agent)
        # shape of obs_next: [batch, num_agents, H, W, C]
        # shape of done: [batch,] (episode termination flag 1 0)

        assert batch.shape[1] == 6, "Batch shape incorrect for low-level policy training."

        # merge n_steps and batch dimensions
        batch = np.asarray(batch) # shape [n_steps, batch, 6] where each i in [6] holds python obj
        batch = batch.reshape(-1, batch.shape[2]) # shape [n_steps * batch, ...]

        # organze batch data
        obs = np.stack(batch[:, 0])
        actions = np.stack(batch[:, 1])
        rewards = np.stack(batch[:, 2])
        skills = np.stack(batch[:, 3])
        obs_next = np.stack(batch[:, 4])
        done = np.stack(batch[:, 5])

        # change them into tensors
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        skills_t = torch.as_tensor(skills, dtype=torch.int64, device=self.device)
        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # encode obs using encoder
        obs_t = self.obs_encoder(obs_t, device=self.device) # change obs_t into shape [n_steps * batch, num_agents, obs_dim]
        obs_next_t = self.obs_encoder(obs_next_t, device=self.device) # change obs_next_t into shape [n_steps *

        # change skills_t into one-hot (num_steps * batch, num_agents, num_skills)
        skills_t = torch.nn.functional.one_hot(skills_t, num_classes=self.num_skills).float()

        # change actions_t into one-hot (num_steps * batch, num_agents, num_actions)
        actions_1hot = torch.nn.functional.one_hot(actions_t, num_classes=self.num_actions).float()

        return batch.shape[0], obs_t, actions_1hot, rewards_t, obs_next_t, skills_t, done_t

    def train_policy_low(self, batch):
        # batch shape: npy list of [policy_obs, actions_int, rewards_low, skills_int, policy_obs_next, done])

        # process batch data
        n_steps, obs, actions_1hot, rewards, obs_next, skills, done = self.process_batch_low(batch)
        
        # train low-level Q-function
        with torch.no_grad(): # one step TD
            q_target = self.Q_low_target(obs_next, skills)
            done_multiplier = -(torch.as_tensor(done, dtype=torch.float32, device=self.device) - 1.0) # if done, future value contribution is zero
            target = torch.as_tensor(rewards, dtype=torch.float32, device=self.device) \
                     + self.gamma * torch.max(q_target, dim=1)[0] * done_multiplier
        q_selected = (self.Q_low(obs, skills) * actions_1hot).sum(dim=1)
    
        loss = self.loss_fn(q_selected, target)
        self.low_opt.zero_grad()
        loss.backward()
        self.low_opt.step()

        networks.soft_update(self.Q_low_target, self.Q_low, self.tau)

    def _downsample_traj(self, obs): # helper function for decoder
        obs_downsampled = obs[:, :: self.traj_skip, :]
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[:, :, : self.obs_truncate_length]
        if self.use_state_difference:
            obs_downsampled = obs_downsampled[:, 1:, :] - obs_downsampled[:, :-1, :]
        assert obs_downsampled.shape[1] == self.traj_length_downsampled
        return obs_downsampled

    def train_decoder(self, dataset):
        # dataset: list of (traj_obs, skills) where
        # traj_obs: np.array of shape (traj_length, H, W, C)
        # skills: np.array of shape (num_skills,)

        # organize dataset
        dataset = np.array(dataset)
        obs = np.stack(dataset[:, 0]) # shape [batch, traj_length, obs_dim]
        skills = np.stack(dataset[:, 1]) # shape [batch, ]

        # change them into tensors
        obs = torch.as_tensor(np.array(obs), dtype=torch.float32, device=self.device)
        skills = torch.as_tensor(np.array(skills), dtype=torch.long, device=self.device)

        # encode obs using encoder (change into shape [batch, traj_length, obs_dim])
        obs = self.obs_encoder(obs, device=self.device)

        # downsample trajectory
        obs_downsampled = self._downsample_traj(obs) # shape [batch, traj_length_downsampled, obs_dim]

        # train decoder
        # API requirement of decoder: [B, traj_length_downsampled, obs_dim] -> logits: [B, N_skills], probs: [B, N_skills]
        logits, probs = self.decoder(obs_downsampled)
        loss = self.ce_loss(logits, skills)
        self.decoder_opt.zero_grad()
        loss.backward()
        self.decoder_opt.step()

        # decoder_probs has shape [batch, N_skills]
        prob = torch.sum(probs * torch.as_tensor(skills, dtype=torch.float32, device=self.device), dim=1)
        expected_prob = prob.mean().item()

        return expected_prob

    def compute_intrinsic_reward(self, agents_traj_obs, skills): # gives decoder classification loss which is used during training
        # agents_traj_obs: np.array of shape (batch, traj_length, H, W, C)
        agents_traj_obs = torch.as_tensor(np.array(agents_traj_obs), dtype=torch.float32, device=self.device)
        
        # encode obs using encoder (change into shape [batch, traj_length, obs_dim])
        with torch.no_grad():
            agents_traj_obs = self.obs_encoder(agents_traj_obs, device=self.device) 

        # downsample trajectory
        traj_t = self._downsample_traj(agents_traj_obs)

        with torch.no_grad():
            # API requirement of decoder: [B, traj_length_downsampled, obs_dim] -> probs: [B, N_skills]
            _, decoder_probs = self.decoder(traj_t)
            prob = torch.sum(decoder_probs * torch.as_tensor(skills, dtype=torch.float32, device=self.device), dim=1)
        return prob.cpu().numpy()
    
    @torch.no_grad()
    def reset(self):
        """Reset internal skill storage."""
        self.current_skills = None


    def save(self, path):
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
    
    def prep_training(self):
        self.decoder.train()
        self.Q_low.train()
        self.agent_main.train()
        self.mixer_main.train()