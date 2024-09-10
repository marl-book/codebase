import random

from einops import rearrange
from gymnasium.spaces import flatdim
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from marlbase.utils.models import MultiAgentSharedNetwork, MultiAgentIndependentNetwork
from marlbase.utils.standardise_stream import RunningMeanStd
from marlbase.utils.utils import compute_nstep_returns


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        layers,
        parameter_sharing,
        use_rnn,
        use_orthogonal_init,
        device,
    ):
        super().__init__()
        hidden_size = list(layers)
        optimizer = getattr(optim, cfg.optimizer)
        lr = cfg.lr

        self.action_space = action_space

        self.n_agents = len(obs_space)
        obs_shape = [flatdim(o) for o in obs_space]
        action_shape = [flatdim(a) for a in action_space]

        if not parameter_sharing:
            self.critic = MultiAgentIndependentNetwork(
                obs_shape, hidden_size, action_shape, use_rnn, use_orthogonal_init
            )
            self.target = MultiAgentIndependentNetwork(
                obs_shape, hidden_size, action_shape, use_rnn, use_orthogonal_init
            )
        else:
            self.critic = MultiAgentSharedNetwork(
                obs_shape,
                hidden_size,
                action_shape,
                parameter_sharing,
                use_rnn,
                use_orthogonal_init,
            )
            self.target = MultiAgentSharedNetwork(
                obs_shape,
                hidden_size,
                action_shape,
                parameter_sharing,
                use_rnn,
                use_orthogonal_init,
            )

        self.soft_update(1.0)
        self.to(device)

        for param in self.target.parameters():
            param.requires_grad = False

        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer_class = optimizer

        self.optimizer = optimizer(self.critic.parameters(), lr=lr)

        self.gamma = cfg.gamma
        self.grad_clip = cfg.grad_clip
        self.device = device
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau
        self.n_steps = cfg.n_steps

        self.updates = 0

        self.standardize_returns = cfg.standardize_returns
        self.ret_ms = RunningMeanStd(shape=(self.n_agents,))

        print(self)

    def forward(self, inputs):
        raise NotImplementedError("Forward not implemented. Use act or update instead!")

    def init_hiddens(self, batch_size):
        return self.critic.init_hiddens(batch_size, self.device)

    def act(self, inputs, hiddens, epsilon):
        with torch.no_grad():
            inputs = [
                torch.tensor(i, device=self.device).view(1, 1, -1) for i in inputs
            ]
            values, hiddens = self.critic(inputs, hiddens)
        if epsilon > random.random():
            actions = self.action_space.sample()
        else:
            actions = [value.argmax(-1).squeeze().cpu().item() for value in values]
        return actions, hiddens

    def _compute_loss(self, batch):
        obss = rearrange(
            torch.stack([batch[f"obs{i}"] for i in range(self.n_agents)]),
            "N B E O -> N E B O",
        )
        actions = rearrange(
            torch.stack([batch[f"act{i}"].long() for i in range(self.n_agents)]),
            "N B E 1 -> E B N 1",
        )
        rewards = rearrange(batch["rew"], "B E N -> E B N")
        dones = rearrange(batch["done"].float(), "B E -> E B 1").repeat(
            1, 1, self.n_agents
        )
        filled = rearrange(batch["filled"], "B E -> E B")

        with torch.no_grad():
            q_tp1_values = torch.stack(self.critic(obss, hiddens=None)[0], dim=-2)
            q_next_states = torch.stack(self.target(obss, hiddens=None)[0], dim=-2)
        all_q_states = torch.stack(self.critic(obss[:, :-1], hiddens=None)[0], dim=-2)

        a_prime = q_tp1_values.argmax(-1)
        target_next_states = q_next_states.gather(-1, a_prime.unsqueeze(-1)).squeeze(-1)

        # returns = compute_nstep_returns(
        #     rewards,
        #     dones,
        #     target_next_states,
        #     self.n_steps,
        #     self.gamma,
        # )
        returns = rewards + self.gamma * target_next_states[1:] * (1 - dones[1:])

        if self.standardize_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean.reshape(1, 1, -1)) / torch.sqrt(
                self.ret_ms.var.reshape(1, 1, -1)
            )

        q_states = all_q_states.gather(-1, actions).squeeze(-1)
        loss = (q_states - returns.detach()).pow(2).mean(-1)
        return (loss * filled).sum() / filled.sum()

    def update(self, batch):
        loss = self._compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimizer.step()
        self.update_target()
        return {"loss": loss.item()}

    def update_target(self):
        if (
            self.target_update_interval_or_tau > 1.0
            and self.updates % self.target_update_interval_or_tau == 0
        ):
            # Hard update
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            # Soft update
            self.soft_update(self.target_update_interval_or_tau)
        self.updates += 1

    def soft_update(self, t):
        source, target = self.critic, self.target
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


class VDNetwork(QNetwork):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        layers,
        parameter_sharing,
        use_rnn,
        use_orthogonal_init,
        device,
    ):
        super().__init__(
            obs_space,
            action_space,
            cfg,
            layers,
            parameter_sharing,
            use_rnn,
            use_orthogonal_init,
            device,
        )
        self.ret_ms = RunningMeanStd(shape=(1,))

    def _compute_loss(self, batch):
        obss = rearrange(
            torch.stack([batch[f"obs{i}"] for i in range(self.n_agents)]),
            "N B E O -> N E B O",
        )
        actions = rearrange(
            torch.stack([batch[f"act{i}"].long() for i in range(self.n_agents)]),
            "N B E 1 -> E B N 1",
        )
        # Get reward of agent 0 --> assume cooperative rewards/ same reward for all agents
        rewards = rearrange(batch["rew"], "B E N -> E B N")[:, :, 0]
        dones = rearrange(batch["done"].float(), "B E -> E B")
        filled = rearrange(batch["filled"], "B E -> E B")

        with torch.no_grad():
            q_tp1_values = torch.stack(self.critic(obss, hiddens=None)[0], dim=-2)
            q_next_states = torch.stack(self.target(obss, hiddens=None)[0], dim=-2)
        all_q_states = torch.stack(self.critic(obss[:, :-1], hiddens=None)[0], dim=-2)

        a_prime = q_tp1_values.argmax(-1)
        target_next_states = (
            q_next_states.gather(-1, a_prime.unsqueeze(-1)).squeeze(-1).sum(-1)
        )

        returns = rewards + self.gamma * target_next_states[1:] * (1 - dones[1:])

        if self.standardize_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean.reshape(1, 1, -1)) / torch.sqrt(
                self.ret_ms.var.reshape(1, 1, -1)
            )

        q_states = all_q_states.gather(-1, actions).squeeze(-1).sum(-1)
        loss = (q_states - returns.detach()).pow(2)
        return (loss * filled).sum() / filled.sum()


class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim, hypernet_layers, hypernet_embed):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim

        self.embed_dim = embed_dim
        self.hypernet_layers = hypernet_layers
        self.hypernet_embed = hypernet_embed

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            hypernet_embed = self.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        else:
            raise Exception(
                "Error setting number of hypernet layers (please set `hypernet_layers=1` or `hypernet_layers=2`)."
            )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        ep_length, batch_size, _ = agent_qs.shape
        agent_qs = rearrange(agent_qs, "E B N -> (E B) 1 N")
        states = states.reshape(-1, self.state_dim)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        return y.view(ep_length, batch_size)


class QMixNetwork(QNetwork):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        layers,
        parameter_sharing,
        use_rnn,
        use_orthogonal_init,
        mixing,
        device,
    ):
        super().__init__(
            obs_space,
            action_space,
            cfg,
            layers,
            parameter_sharing,
            use_rnn,
            use_orthogonal_init,
            device,
        )
        self.ret_ms = RunningMeanStd(shape=(1,))

        state_dim = sum([flatdim(o) for o in obs_space])
        self.mixer = QMixer(self.n_agents, state_dim, **mixing)
        self.target_mixer = QMixer(self.n_agents, state_dim, **mixing)
        self.soft_update(1.0)

        for param in self.target_mixer.parameters():
            param.requires_grad = False

        self.optimizer = self.optimizer_class(
            list(self.critic.parameters()) + list(self.mixer.parameters()),
            lr=cfg.lr,
        )
        print(self)

    def _compute_loss(self, batch):
        obss = rearrange(
            torch.stack([batch[f"obs{i}"] for i in range(self.n_agents)]),
            "N B E O -> N E B O",
        )
        actions = rearrange(
            torch.stack([batch[f"act{i}"].long() for i in range(self.n_agents)]),
            "N B E 1 -> E B N 1",
        )
        # Get reward of agent 0 --> assume cooperative rewards/ same reward for all agents
        rewards = rearrange(batch["rew"], "B E N -> E B N")[:, :, 0]
        dones = rearrange(batch["done"].float(), "B E -> E B")
        filled = rearrange(batch["filled"], "B E -> E B")

        with torch.no_grad():
            q_tp1_values = torch.stack(self.critic(obss, hiddens=None)[0], dim=-2)
            q_next_states = torch.stack(self.target(obss, hiddens=None)[0], dim=-2)
        all_q_states = torch.stack(self.critic(obss[:, :-1], hiddens=None)[0], dim=-2)

        a_prime = q_tp1_values.argmax(-1)
        target_next_states = self.target_mixer(
            q_next_states.gather(-1, a_prime.unsqueeze(-1)).squeeze(-1),
            torch.concat(list(obss), dim=-1),
        )

        returns = rewards + self.gamma * target_next_states[1:] * (1 - dones[1:])

        if self.standardize_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean.reshape(1, 1, -1)) / torch.sqrt(
                self.ret_ms.var.reshape(1, 1, -1)
            )

        q_states = self.mixer(
            all_q_states.gather(-1, actions).squeeze(-1),
            torch.concat(list(obss[:, :-1]), dim=-1),
        )
        loss = (q_states - returns.detach()).pow(2)
        return (loss * filled).sum() / filled.sum()

    def soft_update(self, t):
        super().soft_update(t)
        try:
            source, target = self.mixer, self.target_mixer
        except AttributeError:  # fix for when qmix has not initialised a mixer yet
            return
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)
