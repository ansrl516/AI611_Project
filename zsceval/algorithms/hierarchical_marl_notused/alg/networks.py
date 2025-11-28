import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_layer(layer):
    """Small helper to mirror the narrow TF initialization that was used."""
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, obs_dim, role_dim, n_h1, n_h2, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + role_dim, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.out = nn.Linear(n_h2, n_actions)
        self.apply(_init_layer)

    def forward(self, obs, role):
        x = torch.cat([obs, role], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim, role_dim, n_h1, n_h2):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + role_dim, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.out = nn.Linear(n_h2, 1)
        self.apply(_init_layer)

    def forward(self, obs, role):
        x = torch.cat([obs, role], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class Decoder(nn.Module):
    """Bidirectional LSTM with mean pool over time."""

    def __init__(self, obs_dim, timesteps, n_h=128, n_logits=8):
        super().__init__()
        self.timesteps = timesteps
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=n_h,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.Linear(2 * n_h, n_logits)
        self.apply(_init_layer)

    def forward(self, trajs):
        outputs, _ = self.lstm(trajs)
        pooled = outputs.mean(dim=1)
        logits = self.out(pooled)
        return logits, F.softmax(logits, dim=-1)


class QmixSingle(nn.Module):
    def __init__(self, input_dim, n_h1, n_h2, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.out = nn.Linear(n_h2, n_actions)
        self.apply(_init_layer)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.out(x)


class QLow(nn.Module):
    def __init__(self, obs_dim, role_dim, n_h1, n_h2, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + role_dim, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.out = nn.Linear(n_h2, n_actions)
        self.apply(_init_layer)

    def forward(self, obs, role):
        x = torch.cat([obs, role], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class QHigh(nn.Module):
    def __init__(self, state_dim, n_h1, n_h2, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.out = nn.Linear(n_h2, n_actions)
        self.apply(_init_layer)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)


class QmixMixer(nn.Module):
    def __init__(self, state_dim, n_agents, n_h_mixer):
        super().__init__()
        self.n_agents = n_agents
        self.n_h_mixer = n_h_mixer
        self.hyper_w1 = nn.Linear(state_dim, n_h_mixer * n_agents)
        self.hyper_b1 = nn.Linear(state_dim, n_h_mixer)
        self.hyper_w_final = nn.Linear(state_dim, n_h_mixer)
        self.hyper_b_final_1 = nn.Linear(state_dim, n_h_mixer, bias=False)
        self.hyper_b_final_2 = nn.Linear(n_h_mixer, 1, bias=False)
        self.apply(_init_layer)

    def forward(self, agent_qs, state):
        w1 = torch.abs(self.hyper_w1(state)).view(-1, self.n_agents, self.n_h_mixer)
        b1 = self.hyper_b1(state).view(-1, 1, self.n_h_mixer)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)

        w_final = torch.abs(self.hyper_w_final(state)).view(-1, self.n_h_mixer, 1)
        b_final = self.hyper_b_final_2(F.relu(self.hyper_b_final_1(state))).view(-1, 1, 1)
        y = torch.bmm(hidden, w_final) + b_final
        return y.view(-1, 1)


def soft_update(target, source, tau):
    """Soft-update target network parameters."""
    for targ_param, src_param in zip(target.parameters(), source.parameters()):
        targ_param.data.copy_(tau * src_param.data + (1.0 - tau) * targ_param.data)
