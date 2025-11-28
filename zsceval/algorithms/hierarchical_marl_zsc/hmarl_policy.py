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
        assert param_sharing_option == "all_shared", "Only 'all_shared' parameter sharing option is supported."

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

        # FIXME: Add internal states in Policy? (self.current_skills, etc.)

    def get_actions(self, list_obs, skills, available_actions, epsilon):
        """Get low-level actions for all agents as a batch."""
        # Shape of list_obs: [num_agents, obs_dim] where each entry stands for individual observation of that agent
        # Shape of skills: [num_agents,] where each entry is int skill of that agent
        # Shape of available_actions: [num_agents, num_actions] where each entry is binary mask of available actions
        # Shape of actions: [num_agents] where each entry is int action of that agent 

        # Transform into input form of Q network (Tensor)
        obs = torch.as_tensor(np.array(list_obs), dtype=torch.float32, device=self.device) # shape [num_agents, obs_dim] (obs per agent)
        skills_t = torch.as_tensor(np.array(skills), dtype=torch.float32, device=self.device) # shape [num_agents,] where each entry is int skill of that agent
        skills_t = torch.nn.functional.one_hot(skills_t.long(), num_classes=self.num_skills).float()
        available_actions_t = torch.as_tensor(np.array(available_actions), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q_values = self.Q_low(obs, skills_t) # get Q-values for each low level action given skill
            masked_q = q_values.masked_fill(~available_actions_t, float('-inf'))
            greedy_actions = torch.argmax(masked_q, dim=1).cpu().numpy() # select greedy low-level action per agent within available actions

        actions = np.zeros(self.num_agents, dtype=int)
        for idx in range(self.num_agents):
            if np.random.rand() < epsilon:
                valid_actions = np.where(available_actions[idx] == 1)[0]
                actions[idx] = np.random.choice(valid_actions)
            else:
                actions[idx] = int(greedy_actions[idx])
        return actions

    def get_actions_batch(self, batch_list_obs, batch_skills, batch_available_actions, epsilon):
            """
            Batched low-level action selection.
            
            Args:
                batch_list_obs:        [B, N, obs_dim]
                batch_skills:          [B, N]
                batch_available_actions [B, N, A]
                epsilon: float

            Returns:
                actions: np.ndarray [B, N]
            """
            B, N, obs_dim = batch_list_obs.shape
            A = batch_available_actions.shape[-1]

            # Flatten for network input: (B*N, obs_dim)
            obs = torch.as_tensor(
                batch_list_obs.reshape(B * N, obs_dim),
                dtype=torch.float32, device=self.device
            )

            skills = torch.as_tensor(
                batch_skills.reshape(B * N),
                dtype=torch.long, device=self.device
            )
            skills_1hot = torch.nn.functional.one_hot(
                skills, num_classes=self.num_skills
            ).float()

            avail = torch.as_tensor(
                batch_available_actions.reshape(B * N, A),
                dtype=torch.bool, device=self.device
            )

            with torch.no_grad():
                q_vals = self.Q_low(obs, skills_1hot)             # [B*N, A]
                q_vals = q_vals.masked_fill(~avail, -float("inf"))
                greedy = torch.argmax(q_vals, dim=1).cpu().numpy() # [B*N]

            # ε-greedy exploration
            greedy = greedy.reshape(B, N)
            actions = greedy.copy()

            for b in range(B):
                for i in range(N):
                    if np.random.rand() < epsilon:
                        valid = np.where(batch_available_actions[b, i] == 1)[0]
                        actions[b, i] = np.random.choice(valid)

            return actions

    def assign_skills(self, share_obs, N_skills_current, epsilon=None): 
        """ Assign skills to agents using high-level policy. """
        # share_obs: [num_agents, state_dim] where this is the shared observation available to all agents
        # skills: [num_agents,] where each entry is int skill of that agent
        obs = torch.as_tensor(np.array(share_obs), dtype=torch.float32, device=self.device) # shape [num_agents, obs_dim] (obs per agent)
        with torch.no_grad():
            q_values = self.agent_main(obs)[:, :N_skills_current] # get Q-values for each skill
            skills_argmax = torch.argmax(q_values, dim=1).cpu().numpy() # select greedy skill per agent

        skills = np.zeros(self.num_agents, dtype=int)
        for idx in range(self.num_agents):
            if np.random.rand() < epsilon:
                skills[idx] = np.random.randint(0, N_skills_current)
            else:
                skills[idx] = int(skills_argmax[idx])
        return skills

    def assign_skills_batch(self, batch_share_obs, N_skills_current, epsilon):
        """
        Batched high-level skill assignment.

        Args:
            batch_share_obs:   [B, N, state_dim]
            N_skills_current:  number of skills available
            epsilon:           exploration rate

        Returns:
            skills: np.ndarray [B, N]
        """
        B, N, state_dim = batch_share_obs.shape

        obs = torch.as_tensor(
            batch_share_obs.reshape(B * N, state_dim),
            dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            q_vals = self.agent_main(obs)[:, :N_skills_current]   # [B*N, K]
            greedy = torch.argmax(q_vals, dim=1).cpu().numpy()
            greedy = greedy.reshape(B, N)

        # ε-greedy exploration
        skills = greedy.copy()
        for b in range(B):
            for i in range(N):
                if np.random.rand() < epsilon:
                    skills[b, i] = np.random.randint(0, N_skills_current)

        return skills

    def process_skills(self, n_steps, skills, n_skills): # helper function for train_policy_high, ...
        # Convert skills integer to one-hot encoding
        # shape of skills: [n_steps, num_agents], n_skills: total number of skills
        skills_1hot = np.zeros([n_steps, self.num_agents, n_skills], dtype=int)
        grid = np.indices((n_steps, self.num_agents))
        skills_1hot[grid[0], grid[1], skills] = 1
        skills_1hot.shape = (n_steps * self.num_agents, n_skills)
        return skills_1hot

    def process_batch_high(self, batch): # helper function for high-level policy training
        # batch content: npy list of [share_obs_high, policy_obs_high, skills_int, rewards_high, share_obs, policy_obs, done]
        # shape of share_obs_high: [batch, state_dim]
        # shape of policy_obs_high: [batch, obs_dim] - not used
        # shape of skills_int: [batch,] (int skill per agent)
        # shape of rewards_high: [batch,]
        # shape of share_obs: [batch, state_dim]
        # shape of policy_obs: [batch, obs_dim] - not used
        # shape of done: [batch,] (episode termination flag 1 0)

        assert batch.shape[1] == 7, "Batch shape incorrect for high-level policy training."
        # concat across batch dimension
        state = np.stack(batch[:, 0])
        obs = np.stack(batch[:, 1])
        skills = np.stack(batch[:, 2])
        reward = np.stack(batch[:, 3])
        state_next = np.stack(batch[:, 4])
        obs_next = np.stack(batch[:, 5])
        done = np.stack(batch[:, 6])

        n_steps = state.shape[0]
        obs.shape = (n_steps * self.num_agents, self.obs_dim)
        obs_next.shape = (n_steps * self.num_agents, self.obs_dim)
        skills_1hot = self.process_skills(n_steps, skills, self.num_skills)
        return n_steps, state, obs, skills_1hot, reward, state_next, obs_next, done

    def train_policy_high(self, batch):
        # batch shape: npy list of [share_obs_high, policy_obs_high, skills_int, rewards_high, share_obs, policy_obs, done]
        # process batch data 
        n_steps, state, obs, skills_1hot, reward, state_next, obs_next, done = self.process_batch_high(batch)
        # train high-level Q-function
        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        with torch.no_grad(): # one step TD
            argmax_actions = torch.argmax(self.agent_target(obs_next_t), dim=1)
            skills_target_1hot = torch.zeros(
                (n_steps * self.num_agents, self.num_skills), dtype=torch.float32, device=self.device
            )
            skills_target_1hot.scatter_(1, argmax_actions.unsqueeze(1), 1.0)

            q_target_selected = (self.agent_target(obs_next_t) * skills_target_1hot).sum(dim=1)
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
        # batch content: npy list of [policy_obs_low, actions_int, rewards_low, skills_int, policy_obs_next, done]
        # shape of policy_obs_low: [batch, obs_dim]
        # shape of actions_int: [batch,] (int action per agent)
        # shape of rewards_low: [batch,]
        # shape of skills_int: [batch,] (int skill per agent)
        # shape of policy_obs_next: [batch, obs_dim]
        # shape of done: [batch,] (episode termination flag 1 0)

        # concat across batch dimension
        obs = np.stack(batch[:, 0])
        actions = np.stack(batch[:, 1])
        rewards = np.stack(batch[:, 2])
        obs_next = np.stack(batch[:, 3])
        skills = np.stack(batch[:, 4])
        done = np.stack(batch[:, 5])

        n_steps = obs.shape[0]
        obs.shape = (n_steps * self.num_agents, self.obs_dim)
        obs_next.shape = (n_steps * self.num_agents, self.obs_dim)
        rewards.shape = (n_steps * self.num_agents)
        skills.shape = (n_steps * self.num_agents, self.num_skills)
        done = np.repeat(done, self.num_agents, axis=0)

        actions_1hot = self.process_actions(n_steps, actions, self.num_actions)
        return n_steps, obs, actions_1hot, rewards, obs_next, skills, done

    def train_policy_low(self, batch):
        # batch shape: npy list of [policy_obs, actions_int, rewards_low, skills_int, policy_obs_next, done])

        # process batch data
        n_steps, obs, actions_1hot, rewards, obs_next, skills, done = self.process_batch_low(batch)
        
        # train low-level Q-function
        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        skills_next_t = torch.as_tensor(skills, dtype=torch.float32, device=self.device)
        with torch.no_grad(): # one step TD
            q_target = self.Q_low_target(obs_next_t, skills_next_t)
            done_multiplier = -(torch.as_tensor(done, dtype=torch.float32, device=self.device) - 1.0) # if done, future value contribution is zero
            target = torch.as_tensor(rewards, dtype=torch.float32, device=self.device) \
                     + self.gamma * torch.max(q_target, dim=1)[0] * done_multiplier

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        skills_t = torch.as_tensor(skills, dtype=torch.float32, device=self.device)
        actions_1hot_t = torch.as_tensor(actions_1hot, dtype=torch.float32, device=self.device)
        q_selected = (self.Q_low(obs_t, skills_t) * actions_1hot_t).sum(dim=1)

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
        dataset = np.array(dataset)
        obs = np.stack(dataset[:, 0])
        skills = np.stack(dataset[:, 1])

        obs_downsampled = self._downsample_traj(obs)
        labels = np.argmax(skills, axis=1)

        obs_t = torch.as_tensor(obs_downsampled, dtype=torch.float32, device=self.device)
        labels_t = torch.as_tensor(labels, dtype=torch.long, device=self.device)

        logits, probs = self.decoder(obs_t)
        loss = self.ce_loss(logits, labels_t)
        self.decoder_opt.zero_grad()
        loss.backward()
        self.decoder_opt.step()

        # decoder_probs has shape [batch, N_skills]
        prob = torch.sum(probs * torch.as_tensor(skills, dtype=torch.float32, device=self.device), dim=1)
        expected_prob = prob.mean().item()

        return expected_prob

    def compute_intrinsic_reward(self, agents_traj_obs, skills): # gives decoder classification loss which is used during training
        obs_downsampled = self._downsample_traj(agents_traj_obs)
        traj_t = torch.as_tensor(obs_downsampled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, decoder_probs = self.decoder(traj_t)
            prob = torch.sum(decoder_probs * torch.as_tensor(skills, dtype=torch.float32, device=self.device), dim=1)
        return prob.cpu().numpy()

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
