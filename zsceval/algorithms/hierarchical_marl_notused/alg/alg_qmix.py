"""PyTorch implementation of QMIX."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import networks


def hard_update(target, source):
    for targ_param, src_param in zip(target.parameters(), source.parameters()):
        targ_param.data.copy_(src_param.data)


class Alg(object):
    def __init__(self, config_alg, n_agents, l_state, l_obs, l_action, nn_cfg, device=None):
        """
        Args:
            config_alg: dictionary of general RL params
            n_agents: number of agents on the team controlled by this alg
            l_state, l_obs, l_action: int
            nn_cfg: dictionary with neural net sizes
        """
        self.l_state = l_state
        self.l_obs = l_obs
        self.l_action = l_action
        self.nn = nn_cfg

        self.n_agents = n_agents
        self.tau = config_alg["tau"]
        self.lr_Q = config_alg["lr_Q"]
        self.gamma = config_alg["gamma"]

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.agent_main = networks.QmixSingle(self.l_obs, self.nn["n_h1"], self.nn["n_h2"], self.l_action).to(
            self.device
        )
        self.agent_target = networks.QmixSingle(self.l_obs, self.nn["n_h1"], self.nn["n_h2"], self.l_action).to(
            self.device
        )
        self.mixer_main = networks.QmixMixer(self.l_state, self.n_agents, self.nn["n_h_mixer"]).to(self.device)
        self.mixer_target = networks.QmixMixer(self.l_state, self.n_agents, self.nn["n_h_mixer"]).to(self.device)
        hard_update(self.agent_target, self.agent_main)
        hard_update(self.mixer_target, self.mixer_main)

        self.mixer_opt = optim.Adam(
            list(self.agent_main.parameters()) + list(self.mixer_main.parameters()), lr=self.lr_Q
        )
        self.loss_fn = nn.MSELoss()

    def run_actor(self, list_obs, epsilon):
        """Get actions for all agents as a batch."""
        obs = torch.as_tensor(np.array(list_obs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.agent_main(obs)
            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()

        actions = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[idx] = np.random.randint(0, self.l_action)
            else:
                actions[idx] = int(greedy_actions[idx])
        return actions

    def process_actions(self, n_steps, actions):
        actions_1hot = np.zeros([n_steps, self.n_agents, self.l_action], dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        actions_1hot[grid[0], grid[1], actions] = 1
        actions_1hot.shape = (n_steps * self.n_agents, self.l_action)
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

        actions_1hot = self.process_actions(n_steps, actions)
        return n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done

    def train_step(self, batch, step_train=0, summarize=False, writer=None):
        n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done = self.process_batch(batch)

        obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            argmax_actions = torch.argmax(self.agent_target(obs_next_t), dim=1)
            actions_target_1hot = torch.zeros(
                (n_steps * self.n_agents, self.l_action), dtype=torch.float32, device=self.device
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
        self.mixer_opt.zero_grad()
        loss.backward()
        self.mixer_opt.step()

        networks.soft_update(self.agent_target, self.agent_main, self.tau)
        networks.soft_update(self.mixer_target, self.mixer_main, self.tau)

        if summarize and writer is not None:
            writer.add_scalar("loss_mixer", loss.item(), step_train)

    def save(self, path):
        torch.save(
            {
                "agent_main": self.agent_main.state_dict(),
                "agent_target": self.agent_target.state_dict(),
                "mixer_main": self.mixer_main.state_dict(),
                "mixer_target": self.mixer_target.state_dict(),
            },
            path,
        )

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.agent_main.load_state_dict(checkpoint["agent_main"])
        self.agent_target.load_state_dict(checkpoint["agent_target"])
        self.mixer_main.load_state_dict(checkpoint["mixer_main"])
        self.mixer_target.load_state_dict(checkpoint["mixer_target"])
