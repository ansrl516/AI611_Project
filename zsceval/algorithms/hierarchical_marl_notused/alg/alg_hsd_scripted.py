"""PyTorch implementation of HSD-scripted and MARA-C."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import networks


def hard_update(target, source):
    for targ_param, src_param in zip(target.parameters(), source.parameters()):
        targ_param.data.copy_(src_param.data)


class Alg(object):
    def __init__(self, alg_name, config_alg, n_agents, l_state, l_obs, l_action, N_roles, nn_cfg, device=None):
        """
        Args:
            alg_name: 'hsd-scripted' (QMIX high level) or 'mara-c' (centralized Q-learning)
            config_alg: dictionary of general RL params
            n_agents: number of agents on the team controlled by this alg
            l_state, l_obs, l_action, N_roles: int
            nn_cfg: dictionary with neural net sizes
        """
        self.alg_name = alg_name
        self.l_state = l_state
        self.l_obs = l_obs
        self.l_action = l_action
        self.N_roles = N_roles
        self.nn = nn_cfg

        self.n_agents = n_agents
        self.tau = config_alg["tau"]
        self.lr_Q = config_alg["lr_Q"]
        self.gamma = config_alg["gamma"]

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.alg_name == "mara-c":
            assert self.N_roles >= self.n_agents
            self.dim_role_space = int(np.math.factorial(self.N_roles) / np.math.factorial(self.N_roles - self.n_agents))
            self.list_list_indices = []
            self._populate_list_list_roles(0, [])

        # Low-level Q-functions
        self.Q_low = networks.QLow(self.l_obs, self.N_roles, self.nn["n_h1_low"], self.nn["n_h2_low"], self.l_action).to(self.device)
        self.Q_low_target = networks.QLow(self.l_obs, self.N_roles, self.nn["n_h1_low"], self.nn["n_h2_low"], self.l_action).to(self.device)
        hard_update(self.Q_low_target, self.Q_low)
        self.low_opt = optim.Adam(self.Q_low.parameters(), lr=self.lr_Q)
        self.loss_fn = nn.MSELoss()

        if self.alg_name == "hsd-scripted":
            self.agent_main = networks.QmixSingle(self.l_obs, self.nn["n_h1"], self.nn["n_h2"], self.N_roles).to(self.device)
            self.agent_target = networks.QmixSingle(self.l_obs, self.nn["n_h1"], self.nn["n_h2"], self.N_roles).to(self.device)
            self.mixer_main = networks.QmixMixer(self.l_state, self.n_agents, self.nn["n_h_mixer"]).to(self.device)
            self.mixer_target = networks.QmixMixer(self.l_state, self.n_agents, self.nn["n_h_mixer"]).to(self.device)
            hard_update(self.agent_target, self.agent_main)
            hard_update(self.mixer_target, self.mixer_main)
            self.high_opt = optim.Adam(list(self.agent_main.parameters()) + list(self.mixer_main.parameters()), lr=self.lr_Q)
        elif self.alg_name == "mara-c":
            self.Q_high = networks.QHigh(self.l_state, self.nn["n_h1"], self.nn["n_h2"], self.dim_role_space).to(
                self.device
            )
            self.Q_high_target = networks.QHigh(
                self.l_state, self.nn["n_h1"], self.nn["n_h2"], self.dim_role_space
            ).to(self.device)
            hard_update(self.Q_high_target, self.Q_high)
            self.high_opt = optim.Adam(self.Q_high.parameters(), lr=self.lr_Q)

    def _populate_list_list_roles(self, agent_idx, list_indices): # enlists permutations (generates lists of roles index s.t it is unique across agents)
        if len(list_indices) == self.n_agents:
            self.list_list_indices.append(list_indices)
        else:
            for idx_role in range(self.N_roles):
                if idx_role in list_indices:
                    continue
                l = list(list_indices)
                l.append(idx_role)
                self._populate_list_list_roles(agent_idx + 1, l)

    def run_actor(self, list_obs, roles, epsilon): # receives roles assigned from high-level policy (assign_roles or assign_roles_centralized)
        obs = torch.as_tensor(np.array(list_obs), dtype=torch.float32, device=self.device) # observations for each agent
        roles_t = torch.as_tensor(np.array(roles), dtype=torch.float32, device=self.device) # roles for each agent
        with torch.no_grad(): # selecting actions does not require gradient calculation
            q_values = self.Q_low(obs, roles_t) # get Q-values for low-level actions
            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy() # from that list of low-level actions, perform greedily

        actions = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand() < epsilon: # if exploration, perform random action
                actions[idx] = np.random.randint(0, self.l_action)
            else: # else, perform exploitation
                actions[idx] = int(greedy_actions[idx])
        return actions

    def assign_roles(self, list_obs, epsilon): # decentralized role assignment
        obs = torch.as_tensor(np.array(list_obs), dtype=torch.float32, device=self.device) # observations for each agent
        with torch.no_grad(): # selecting roles does not require gradient calculation
            q_values = self.agent_main(obs) # prints q_values according to each discrete skill action for each agent
            roles_argmax = torch.argmax(q_values, dim=1).cpu().numpy() # select best skill for each agent

        roles = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand() < epsilon:
                roles[idx] = np.random.randint(0, self.N_roles)
            else:
                roles[idx] = int(roles_argmax[idx])
        return roles

    def assign_roles_centralized(self, state, epsilon): # centralized role assignment
        state_t = torch.as_tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0) # whole observation of the team, ...
        with torch.no_grad():
            q_values = self.Q_high(state_t).squeeze(0) # Q-values for joint role assignments for each agent
            idx_action_argmax = int(torch.argmax(q_values).item()) # select best joint role assignment
        if np.random.rand() < epsilon:
            idx_action = np.random.randint(0, self.dim_role_space)
        else:
            idx_action = idx_action_argmax
        roles = np.array(self.list_list_indices[idx_action])
        return roles, idx_action

    def process_actions(self, n_steps, actions, n_actions): # groups actions in shape expected by the network
        actions_1hot = np.zeros([n_steps, self.n_agents, n_actions], dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        actions_1hot[grid[0], grid[1], actions] = 1
        actions_1hot.shape = (n_steps * self.n_agents, n_actions)
        return actions_1hot

    def process_batch(self, batch): # process batch for high-level training
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
        actions_1hot = self.process_actions(n_steps, actions, self.N_roles)
        return n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done # collects trajectory to train Q functions

    def train_step(self, batch, step_train=0, summarize=False, writer=None): # high-level Q training step
        if self.alg_name == "hsd-scripted":
            n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done = self.process_batch(batch) # collect trajectory to train Q functions

            # Algorithm of QMIX high-level (centralized mixer update using decentralized agent Q-values)
            obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                argmax_actions = torch.argmax(self.agent_target(obs_next_t), dim=1)
                actions_target_1hot = torch.zeros(
                    (n_steps * self.n_agents, self.N_roles), dtype=torch.float32, device=self.device
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
        elif self.alg_name == "mara-c":
            state = np.stack(batch[:, 0])
            actions = np.stack(batch[:, 1])
            reward = np.stack(batch[:, 2])
            state_next = np.stack(batch[:, 3])
            done = np.stack(batch[:, 4])

            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            actions_1hot = torch.zeros((len(actions), self.dim_role_space), dtype=torch.float32, device=self.device)
            actions_1hot.scatter_(1, torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1), 1.0)

            with torch.no_grad():
                state_next_t = torch.as_tensor(state_next, dtype=torch.float32, device=self.device)
                q_target = self.Q_high_target(state_next_t)
                done_multiplier = -(torch.as_tensor(done, dtype=torch.float32, device=self.device) - 1.0)
                target = torch.as_tensor(reward, dtype=torch.float32, device=self.device) + self.gamma * torch.max(
                    q_target, dim=1
                )[0] * done_multiplier

            q_pred = (self.Q_high(state_t) * actions_1hot).sum(dim=1)
            loss = self.loss_fn(q_pred, target)
            self.high_opt.zero_grad()
            loss.backward()
            self.high_opt.step()

            networks.soft_update(self.Q_high_target, self.Q_high, self.tau)

            if summarize and writer is not None:
                writer.add_scalar("loss_Q_high", loss.item(), step_train)

    def process_batch_low(self, batch): # process batch for low-level training
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
        roles.shape = (n_steps * self.n_agents, self.N_roles)
        done = np.repeat(done, self.n_agents, axis=0)

        actions_1hot = self.process_actions(n_steps, actions, self.l_action)
        return n_steps, obs, actions_1hot, rewards, obs_next, roles, done

    def train_step_low(self, batch, step_train=0, summarize=False, writer=None): # low-level Q training step
        n_steps, obs, actions_1hot, rewards, obs_next, roles, done = self.process_batch_low(batch)

        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        roles_next_t = torch.as_tensor(roles, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_target = self.Q_low_target(obs_next_t, roles_next_t)
            done_multiplier = -(torch.as_tensor(done, dtype=torch.float32, device=self.device) - 1.0)
            target = torch.as_tensor(rewards, dtype=torch.float32, device=self.device) + self.gamma * torch.max(
                q_target, dim=1
            )[0] * done_multiplier

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

    def save(self, path):
        payload = {
            "Q_low": self.Q_low.state_dict(),
            "Q_low_target": self.Q_low_target.state_dict(),
        }
        if self.alg_name == "hsd-scripted":
            payload.update(
                {
                    "agent_main": self.agent_main.state_dict(),
                    "agent_target": self.agent_target.state_dict(),
                    "mixer_main": self.mixer_main.state_dict(),
                    "mixer_target": self.mixer_target.state_dict(),
                }
            )
        elif self.alg_name == "mara-c":
            payload.update({"Q_high": self.Q_high.state_dict(), "Q_high_target": self.Q_high_target.state_dict()})
        torch.save(payload, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.Q_low.load_state_dict(checkpoint["Q_low"])
        self.Q_low_target.load_state_dict(checkpoint["Q_low_target"])
        if self.alg_name == "hsd-scripted":
            self.agent_main.load_state_dict(checkpoint["agent_main"])
            self.agent_target.load_state_dict(checkpoint["agent_target"])
            self.mixer_main.load_state_dict(checkpoint["mixer_main"])
            self.mixer_target.load_state_dict(checkpoint["mixer_target"])
        elif self.alg_name == "mara-c":
            self.Q_high.load_state_dict(checkpoint["Q_high"])
            self.Q_high_target.load_state_dict(checkpoint["Q_high_target"])
