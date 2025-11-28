"""PyTorch implementation of hierarchical cooperative MARL with skill discovery (HSD)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import networks


def hard_update(target, source):
    for targ_param, src_param in zip(target.parameters(), source.parameters()):
        targ_param.data.copy_(src_param.data)


class Alg(object):
    def __init__(self, config_alg, config_h, n_agents, l_state, l_obs, l_action, l_z, nn_cfg, device=None):
        """
        Args:
            config_alg: dictionary of general RL params
            config_h: dictionary of HSD params
            n_agents: number of agents on the team controlled by this alg
            l_state, l_obs, l_action, l_z: int
            nn_cfg: dictionary with neural net sizes
        """
        self.l_state = l_state
        self.l_obs = l_obs
        self.l_action = l_action
        self.l_z = l_z
        self.nn = nn_cfg

        self.n_agents = n_agents
        self.tau = config_alg["tau"]
        self.lr_Q = config_alg["lr_Q"]
        self.lr_actor = config_alg["lr_actor"]
        self.lr_decoder = config_alg["lr_decoder"]
        self.gamma = config_alg["gamma"]

        self.traj_length = config_h["steps_per_assign"]
        self.traj_skip = config_h["traj_skip"]
        self.traj_length_downsampled = int(np.ceil(self.traj_length / self.traj_skip))
        self.use_state_difference = config_h["use_state_difference"]
        if self.use_state_difference:
            self.traj_length_downsampled -= 1

        self.obs_truncate_length = config_h["obs_truncate_length"]
        assert (self.obs_truncate_length is None) or (self.obs_truncate_length <= self.l_obs)

        self.low_level_alg = config_h["low_level_alg"]
        if self.low_level_alg != "iql":
            raise NotImplementedError("Only low_level_alg == 'iql' is supported in the PyTorch port.")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        decoder_input_dim = self.obs_truncate_length or self.l_obs
        self.decoder = networks.Decoder(decoder_input_dim, self.traj_length_downsampled, self.nn["n_h_decoder"], self.l_z).to(self.device)
        self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=self.lr_decoder)
        self.ce_loss = nn.CrossEntropyLoss()

        # Low-level Q-functions
        self.Q_low = networks.QLow(self.l_obs, self.l_z, self.nn["n_h1_low"], self.nn["n_h2_low"], self.l_action).to(self.device)
        self.Q_low_target = networks.QLow(self.l_obs, self.l_z, self.nn["n_h1_low"], self.nn["n_h2_low"], self.l_action).to(self.device)
        hard_update(self.Q_low_target, self.Q_low)
        self.low_opt = optim.Adam(self.Q_low.parameters(), lr=self.lr_Q)

        # High-level QMIX
        self.agent_main = networks.QmixSingle(self.l_obs, self.nn["n_h1"], self.nn["n_h2"], self.l_z).to(self.device)
        self.agent_target = networks.QmixSingle(self.l_obs, self.nn["n_h1"], self.nn["n_h2"], self.l_z).to(self.device)
        self.mixer_main = networks.QmixMixer(self.l_state, self.n_agents, self.nn["n_h_mixer"]).to(self.device)
        self.mixer_target = networks.QmixMixer(self.l_state, self.n_agents, self.nn["n_h_mixer"]).to(self.device)
        hard_update(self.agent_target, self.agent_main)
        hard_update(self.mixer_target, self.mixer_main)

        self.high_opt = optim.Adam(list(self.agent_main.parameters()) + list(self.mixer_main.parameters()), lr=self.lr_Q)
        self.loss_fn = nn.MSELoss()

    def run_actor(self, list_obs, roles, epsilon):
        """Get low-level actions for all agents as a batch."""
        obs = torch.as_tensor(np.array(list_obs), dtype=torch.float32, device=self.device) # shape [n_agents, l_obs] (obs per agent)
        roles_t = torch.as_tensor(np.array(roles), dtype=torch.float32, device=self.device) # shape [n_agents, l_z] (roles per agent assigned by high-level policy)
        with torch.no_grad():
            q_values = self.Q_low(obs, roles_t) # get Q-values for each low level action given role
            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy() # select greedy low-level action per agent

        actions = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[idx] = np.random.randint(0, self.l_action)
            else:
                actions[idx] = int(greedy_actions[idx])
        return actions

    def assign_roles(self, list_obs, epsilon, N_roles_current): # Assign roles to agents using high-level policy
        obs = torch.as_tensor(np.array(list_obs), dtype=torch.float32, device=self.device) # shape [n_agents, l_obs] (obs per agent)
        with torch.no_grad():
            q_values = self.agent_main(obs)[:, :N_roles_current] # get Q-values for each role
            roles_argmax = torch.argmax(q_values, dim=1).cpu().numpy() # select greedy role per agent

        roles = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand() < epsilon:
                roles[idx] = np.random.randint(0, N_roles_current)
            else:
                roles[idx] = int(roles_argmax[idx])
        return roles

    def process_actions(self, n_steps, actions, n_actions): # Convert actions to one-hot encoding
        actions_1hot = np.zeros([n_steps, self.n_agents, n_actions], dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        actions_1hot[grid[0], grid[1], actions] = 1
        actions_1hot.shape = (n_steps * self.n_agents, n_actions)
        return actions_1hot

    def process_batch(self, batch):
        state = np.stack(batch[:, 0])
        obs = np.stack(batch[:, 1])
        actions = np.stack(batch[:, 2])
        reward = np.stack(batch[:, 3])
        state_next = np.stack(batch[:, 4])
        obs_next = np.stack(batch[:, 5])
        done = np.stack(batch[:, 6])

        n_steps = state.shape[0]
        obs.shape = (n_steps * self.n_agents, self.l_obs)
        obs_next.shape = (n_steps * self.n_agents, self.l_obs)
        actions_1hot = self.process_actions(n_steps, actions, self.l_z)
        return n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done

    def train_policy_high(self, batch, step_train, summarize=False, writer=None):
        n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done = self.process_batch(batch)

        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        with torch.no_grad(): # one step TD
            argmax_actions = torch.argmax(self.agent_target(obs_next_t), dim=1)
            actions_target_1hot = torch.zeros(
                (n_steps * self.n_agents, self.l_z), dtype=torch.float32, device=self.device
            )
            actions_target_1hot.scatter_(1, argmax_actions.unsqueeze(1), 1.0)

            q_target_selected = (self.agent_target(obs_next_t) * actions_target_1hot).sum(dim=1)
            mixer_input_target = q_target_selected.view(-1, self.n_agents)

            state_next_t = torch.as_tensor(state_next, dtype=torch.float32, device=self.device)
            q_tot_target = self.mixer_target(mixer_input_target, state_next_t).squeeze(1)

            done_multiplier = -(torch.as_tensor(done, dtype=torch.float32, device=self.device) - 1.0)
            target = torch.as_tensor(reward, dtype=torch.float32, device=self.device) + self.gamma * q_tot_target * done_multiplier

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_1hot_t = torch.as_tensor(actions_1hot, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        q_selected = (self.agent_main(obs_t) * actions_1hot_t).sum(dim=1)
        mixer_input = q_selected.view(-1, self.n_agents)
        q_tot = self.mixer_main(mixer_input, state_t).squeeze(1)

        loss = self.loss_fn(q_tot, target)
        self.high_opt.zero_grad()
        loss.backward()
        self.high_opt.step()

        networks.soft_update(self.agent_target, self.agent_main, self.tau)
        networks.soft_update(self.mixer_target, self.mixer_main, self.tau)

        if summarize and writer is not None:
            writer.add_scalar("loss_Q_high", loss.item(), step_train)

    def process_batch_low(self, batch):
        obs = np.stack(batch[:, 0])
        actions = np.stack(batch[:, 1])
        rewards = np.stack(batch[:, 2])
        obs_next = np.stack(batch[:, 3])
        roles = np.stack(batch[:, 4])
        done = np.stack(batch[:, 5])

        n_steps = obs.shape[0]
        obs.shape = (n_steps * self.n_agents, self.l_obs)
        obs_next.shape = (n_steps * self.n_agents, self.l_obs)
        rewards.shape = (n_steps * self.n_agents)
        roles.shape = (n_steps * self.n_agents, self.l_z)
        done = np.repeat(done, self.n_agents, axis=0)

        actions_1hot = self.process_actions(n_steps, actions, self.l_action)
        return n_steps, obs, actions_1hot, rewards, obs_next, roles, done

    def train_policy_low(self, batch, step_train=0, summarize=False, writer=None):
        n_steps, obs, actions_1hot, rewards, obs_next, roles, done = self.process_batch_low(batch)

        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        roles_next_t = torch.as_tensor(roles, dtype=torch.float32, device=self.device)
        with torch.no_grad(): # one step TD
            q_target = self.Q_low_target(obs_next_t, roles_next_t)
            done_multiplier = -(torch.as_tensor(done, dtype=torch.float32, device=self.device) - 1.0) # if done, future value contribution is zero
            target = torch.as_tensor(rewards, dtype=torch.float32, device=self.device) \
                     + self.gamma * torch.max(q_target, dim=1)[0] * done_multiplier

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        roles_t = torch.as_tensor(roles, dtype=torch.float32, device=self.device)
        actions_1hot_t = torch.as_tensor(actions_1hot, dtype=torch.float32, device=self.device)
        q_selected = (self.Q_low(obs_t, roles_t) * actions_1hot_t).sum(dim=1)

        loss = self.loss_fn(q_selected, target)
        self.low_opt.zero_grad()
        loss.backward()
        self.low_opt.step()

        networks.soft_update(self.Q_low_target, self.Q_low, self.tau)

        if summarize and writer is not None:
            writer.add_scalar("loss_IQL_low", loss.item(), step_train)

    def _downsample_traj(self, obs):
        obs_downsampled = obs[:, :: self.traj_skip, :]
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[:, :, : self.obs_truncate_length]
        if self.use_state_difference:
            obs_downsampled = obs_downsampled[:, 1:, :] - obs_downsampled[:, :-1, :]
        assert obs_downsampled.shape[1] == self.traj_length_downsampled
        return obs_downsampled

    def train_decoder(self, dataset, step_train, summarize=False, writer=None):
        dataset = np.array(dataset)
        obs = np.stack(dataset[:, 0])
        z = np.stack(dataset[:, 1])

        obs_downsampled = self._downsample_traj(obs)
        labels = np.argmax(z, axis=1)

        obs_t = torch.as_tensor(obs_downsampled, dtype=torch.float32, device=self.device)
        labels_t = torch.as_tensor(labels, dtype=torch.long, device=self.device)

        logits, probs = self.decoder(obs_t)
        loss = self.ce_loss(logits, labels_t)
        self.decoder_opt.zero_grad()
        loss.backward()
        self.decoder_opt.step()

        # decoder_probs has shape [batch, N_roles]
        prob = torch.sum(probs * torch.as_tensor(z, dtype=torch.float32, device=self.device), dim=1)
        expected_prob = prob.mean().item()

        if summarize and writer is not None:
            writer.add_scalar("decoder_loss", loss.item(), step_train)
            writer.add_scalar("decoder_expected_prob", expected_prob, step_train)

        return expected_prob

    def compute_reward(self, agents_traj_obs, z): # gives decoder classification loss (used in train_hsd.py)
        obs_downsampled = self._downsample_traj(agents_traj_obs)
        traj_t = torch.as_tensor(obs_downsampled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, decoder_probs = self.decoder(traj_t)
            prob = torch.sum(decoder_probs * torch.as_tensor(z, dtype=torch.float32, device=self.device), dim=1)
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
