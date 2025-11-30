import torch
import torch.nn as nn
import torch.nn.functional as F

# Changed: All of the networks below now support both single-agent and batched multi-agent inputs.

def _init_layer(layer):
    """Small helper to mirror the narrow TF initialization that was used."""
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)

class ObsEncoder(nn.Module):
    """
    Option1: torch.tensor of shape (num_agents, H, W, C) -> (num_agents, obs_dim)
    Option2: torch.tensor of shape (n_rollout_threads, num_agents, C, H, W) -> (n_rollout_threads, num_agents, obs_dim)
    """
    def __init__(self, in_channels, obs_embedding_dim=128, H=5, W=5):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Use dummy tensor to compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            flatten_dim = self.cnn(dummy).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_embedding_dim),
            nn.ReLU()
        )

        self.apply(_init_layer)

    def forward(self, obs):
        """
        obs can be:
            (N, H, W, C)
            (B, N, H, W, C)
        """

        if obs.dim() == 4:
            # Case 1: (N, H, W, C)
            # Permute to (N, C, H, W)
            x = obs.permute(0, 3, 1, 2)
            x = self.cnn(x)
            x = self.fc(x)
            return x

        elif obs.dim() == 5:
            # Case 2: (B, N, H, W, C)
            B, N, H, W, C = obs.shape

            # Merge B and N → (B*N, H, W, C)
            x = obs.reshape(B * N, H, W, C)

            # Permute to (B*N, C, H, W)
            x = x.permute(0, 3, 1, 2)

            # Forward CNN + MLP
            x = self.cnn(x)      # (B*N, F)
            x = self.fc(x)       # (B*N, obs_dim)

            # Restore batch structure → (B, N, obs_dim)
            x = x.reshape(B, N, -1)
            return x

        else:
            raise ValueError(
                f"ObsEncoder expected 4D or 5D tensor, got shape {obs.shape}"
            )


class ShareObsEncoder(nn.Module):
    """
    Option1: torch.tensor of shape (num_agents, H, W, C_share) -> (num_agents, state_dim)
    Option2: torch.tensor of shape (n_rollout_threads, num_agents, C_share, H, W) -> (n_rollout_threads, num_agents, state_dim)
    """
    def __init__(self, in_channels, state_embedding_dim=128, H=5, W=5):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            flatten_dim = self.cnn(dummy).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_embedding_dim),
            nn.ReLU()
        )

        self.apply(_init_layer)

    def forward(self, state):
        """
        state can be:
            (N, H, W, C_share)
            (B, N, H, W, C_share)
        """

        if state.dim() == 4:
            # Case 1: (N, H, W, C_share)
            x = state.permute(0, 3, 1, 2)  # (N, C_share, H, W)
            x = self.cnn(x)
            x = self.fc(x)
            return x

        elif state.dim() == 5:
            # Case 2: (B, N, H, W, C_share)
            B, N, H, W, C = state.shape

            # Merge batch and agent dims: (B*N, H, W, C)
            x = state.reshape(B * N, H, W, C)

            # Permute to Conv2D format
            x = x.permute(0, 3, 1, 2)  # (B*N, C, H, W)

            # CNN + MLP
            x = self.cnn(x)           # (B*N, F)
            x = self.fc(x)            # (B*N, state_embedding_dim)

            # Restore batch dimension
            x = x.reshape(B, N, -1)   # (B, N, state_dim)
            return x

        else:
            raise ValueError(
                f"ShareObsEncoder expected 4D or 5D tensor, got {state.shape}"
            )

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
        """
        obs shapes supported:
            (N, input_dim)
            (B, N, input_dim)
        """

        if obs.dim() == 2:
            # (N, F)
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            return self.out(x)

        elif obs.dim() == 3:
            # (B, N, F)
            B, N, F = obs.shape
            x = obs.reshape(B * N, F)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.out(x)
            return x.reshape(B, N, -1)

        else:
            raise ValueError(f"Unsupported obs shape {obs.shape}")


class QLow(nn.Module):
    def __init__(self, obs_dim, role_dim, n_h1, n_h2, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + role_dim, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.out = nn.Linear(n_h2, n_actions)
        self.apply(_init_layer)

    def forward(self, obs, role):
        """
        obs  : (N, obs_dim)       OR (B, N, obs_dim)
        role : (N, role_dim)      OR (B, N, role_dim)
        """

        if obs.dim() == 2:
            # (N, F_obs), (N, F_role)
            x = torch.cat([obs, role], dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.out(x)

        elif obs.dim() == 3:
            # (B, N, F_obs), (B, N, F_role)
            B, N, F_obs = obs.shape
            _, _, F_role = role.shape

            x = torch.cat([obs, role], dim=2)  # (B, N, F_obs+F_role)
            x = x.reshape(B * N, F_obs + F_role)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.out(x)

            return x.reshape(B, N, -1)

        else:
            raise ValueError(f"Unsupported obs/role shape {obs.shape}")

class QHigh(nn.Module):
    def __init__(self, state_dim, n_h1, n_h2, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.out = nn.Linear(n_h2, n_actions)
        self.apply(_init_layer)

    def forward(self, state):
        """
        state: (N, state_dim)
               (B, N, state_dim)
        """

        if state.dim() == 2:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.out(x)

        elif state.dim() == 3:
            B, N, F = state.shape
            x = state.reshape(B * N, F)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.out(x)
            return x.reshape(B, N, -1)

        else:
            raise ValueError(f"Unsupported state shape {state.shape}")
        
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
        """
        agent_qs: (N, n_agents) or (B, N, n_agents)
        state:    (N, F_s)     or (B, N, F_s)
        """

        if agent_qs.dim() == 2:
            # (N, n_agents)
            return self._mix(agent_qs, state)

        elif agent_qs.dim() == 3:
            # (B, N, n_agents)
            B, N, A = agent_qs.shape
            agent_qs_flat = agent_qs.reshape(B * N, A)
            state_flat = state.reshape(B * N, -1)

            y = self._mix(agent_qs_flat, state_flat)   # (B*N, 1)

            return y.reshape(B, N, 1)

        else:
            raise ValueError(f"Unsupported agent_qs shape {agent_qs.shape}")

    def _mix(self, agent_qs, state):
        """
        Core QMIX mixing for flattened batch:
        agent_qs: (BN, n_agents)
        state:    (BN, state_dim)
        """
        w1 = torch.abs(self.hyper_w1(state)).view(-1, self.n_agents, self.n_h_mixer)
        b1 = self.hyper_b1(state).view(-1, 1, self.n_h_mixer)

        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)

        w_final = torch.abs(self.hyper_w_final(state)).view(-1, self.n_h_mixer, 1)
        b_final = self.hyper_b_final_2(F.relu(self.hyper_b_final_1(state))).view(-1, 1, 1)

        y = torch.bmm(hidden, w_final) + b_final   # (BN, 1, 1)
        return y.view(-1, 1)

def soft_update(target, source, tau):
    """Soft-update target network parameters."""
    for targ_param, src_param in zip(target.parameters(), source.parameters()):
        targ_param.data.copy_(tau * src_param.data + (1.0 - tau) * targ_param.data)
