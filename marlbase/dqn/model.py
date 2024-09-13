import random

from einops import rearrange
from gymnasium.spaces import flatdim
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from marlbase.utils.models import MultiAgentSharedNetwork, MultiAgentIndependentNetwork
from marlbase.utils.standardise_stream import RunningMeanStd


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
        hidden_dims = list(layers)

        self.action_space = action_space

        self.n_agents = len(obs_space)
        obs_shape = [flatdim(o) for o in obs_space]
        action_shape = [flatdim(a) for a in action_space]

        if not parameter_sharing:
            self.critic = MultiAgentIndependentNetwork(
                obs_shape, hidden_dims, action_shape, use_rnn, use_orthogonal_init
            )
            self.target = MultiAgentIndependentNetwork(
                obs_shape, hidden_dims, action_shape, use_rnn, use_orthogonal_init
            )
        else:
            self.critic = MultiAgentSharedNetwork(
                obs_shape,
                hidden_dims,
                action_shape,
                parameter_sharing,
                use_rnn,
                use_orthogonal_init,
            )
            self.target = MultiAgentSharedNetwork(
                obs_shape,
                hidden_dims,
                action_shape,
                parameter_sharing,
                use_rnn,
                use_orthogonal_init,
            )

        self.hard_update()
        self.to(device)

        for param in self.target.parameters():
            param.requires_grad = False

        if type(cfg.optimizer) is str:
            self.optimizer_class = getattr(optim, cfg.optimizer)
        else:
            self.optimizer_class = cfg.optimizer

        self.optimizer = self.optimizer_class(self.critic.parameters(), lr=cfg.lr)

        self.gamma = cfg.gamma
        self.grad_clip = cfg.grad_clip
        self.device = device
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau
        self.double_q = cfg.double_q

        self.updates = 0
        self.last_target_update = 0

        self.standardise_returns = cfg.standardise_returns
        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,))
        self.standardise_rewards = cfg.standardise_rewards
        if self.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(self.n_agents,))

        print(self)

    def forward(self, inputs):
        raise NotImplementedError("Forward not implemented. Use act or update instead!")

    def init_hiddens(self, batch_size):
        return self.critic.init_hiddens(batch_size, self.device)

    def act(self, inputs, hiddens, epsilon, action_masks=None):
        with torch.no_grad():
            inputs = [
                torch.tensor(i, device=self.device).view(1, 1, -1) for i in inputs
            ]
            values, hiddens = self.critic(inputs, hiddens)
        if action_masks is not None:
            masked_values = []
            for value, mask in zip(values, action_masks):
                masked_values.append(value * mask + (1 - mask) * -1e8)
            values = masked_values
        if epsilon > random.random():
            if action_masks is not None:
                # random index of action with mask = 1
                actions = [
                    random.choice([i for i, m in enumerate(mask) if m == 1])
                    for mask in action_masks
                ]
            else:
                actions = self.action_space.sample()
        else:
            actions = [value.argmax(-1).squeeze().cpu().item() for value in values]
        return actions, hiddens

    def _compute_loss(self, batch):
        obss = batch.obss
        actions = batch.actions.unsqueeze(-1)
        rewards = batch.rewards
        dones = batch.dones[1:].unsqueeze(0).repeat(self.n_agents, 1, 1)
        filled = batch.filled
        action_masks = batch.action_mask

        if self.standardise_rewards:
            rewards = rearrange(rewards, "N E B -> E B N")
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)
            rewards = rearrange(rewards, "E B N -> N E B")

        # (n_agents, ep_length, batch_size, n_actions)
        q_values, _ = self.critic(obss, hiddens=None)
        q_values = torch.stack(q_values)
        chosen_q_values = q_values[:, :-1].gather(-1, actions).squeeze(-1)

        # compute target
        with torch.no_grad():
            target_q_values, _ = self.target(obss, hiddens=None)
            target_q_values = torch.stack(target_q_values)[:, 1:]
            if action_masks is not None:
                target_q_values[action_masks[:, 1:] == 0] = -1e8

        if self.double_q:
            q_values_clone = q_values.clone().detach()[:, 1:]
            if action_masks is not None:
                q_values_clone[action_masks[:, 1:] == 0] = -1e8
            a_prime = q_values_clone.argmax(-1)
            target_qs = target_q_values.gather(-1, a_prime.unsqueeze(-1)).squeeze(-1)
        else:
            target_qs, _ = target_q_values.max(dim=-1)

        returns = rewards + self.gamma * target_qs.detach() * (1 - dones)

        if self.standardise_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)

        loss = torch.nn.functional.mse_loss(
            chosen_q_values, returns.detach(), reduction="none"
        ).sum(dim=0)
        return (loss * filled).sum() / filled.sum()

    def update(self, batch):
        loss = self._compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimizer.step()
        self.updates += 1
        self.update_target()
        return {"loss": loss.item()}

    def update_target(self):
        if (
            self.target_update_interval_or_tau > 1.0
            and (self.updates - self.last_target_update)
            >= self.target_update_interval_or_tau
        ):
            self.hard_update()
            self.last_target_update = self.updates
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

    def soft_update(self, tau):
        for target_param, source_param in zip(
            self.target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                (1 - tau) * target_param.data + tau * source_param.data
            )

    def hard_update(self):
        self.target.load_state_dict(self.critic.state_dict())


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
        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(1,))
        if self.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,))

    def _compute_loss(self, batch):
        obss = batch.obss
        actions = batch.actions.unsqueeze(-1)
        # Get reward of agent 0 --> assume cooperative rewards/ same reward for all agents
        rewards = batch.rewards[0]
        dones = batch.dones[1:]
        filled = batch.filled
        action_masks = batch.action_mask

        if self.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)

        # (n_agents, ep_length, batch_size, n_actions)
        q_values, _ = self.critic(obss, hiddens=None)
        q_values = torch.stack(q_values)
        # sum over all agents for cooperative VDN estimate
        chosen_q_values = q_values[:, :-1].gather(-1, actions).squeeze(-1).sum(dim=0)

        # compute target
        with torch.no_grad():
            target_q_values, _ = self.target(obss, hiddens=None)
            target_q_values = torch.stack(target_q_values)[:, 1:]
            if action_masks is not None:
                target_q_values[action_masks[:, 1:] == 0] = -1e8

        if self.double_q:
            q_values_clone = q_values.clone().detach()[:, 1:]
            if action_masks is not None:
                q_values_clone[action_masks[:, 1:] == 0] = -1e8
            a_prime = q_values_clone.argmax(-1)
            target_qs = target_q_values.gather(-1, a_prime.unsqueeze(-1)).squeeze(-1)
        else:
            target_qs, _ = target_q_values.max(dim=-1)

        # sum over target values of all agents for cooperative VDN target
        returns = rewards + self.gamma * target_qs.detach().sum(dim=0) * (1 - dones)

        if self.standardise_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)

        loss = torch.nn.functional.mse_loss(
            chosen_q_values, returns.detach(), reduction="none"
        )
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
        _, ep_length, batch_size = agent_qs.shape
        agent_qs = rearrange(agent_qs, "N E B -> (E B) 1 N")
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
        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(1,))
        if self.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,))

        state_dim = sum([flatdim(o) for o in obs_space])
        self.mixer = QMixer(self.n_agents, state_dim, **mixing)
        self.target_mixer = QMixer(self.n_agents, state_dim, **mixing)
        self.hard_update()

        for param in self.target_mixer.parameters():
            param.requires_grad = False

        self.optimizer = self.optimizer_class(
            list(self.critic.parameters()) + list(self.mixer.parameters()),
            lr=cfg.lr,
        )
        print(self)

    def _compute_loss(self, batch):
        obss = batch.obss
        actions = batch.actions.unsqueeze(-1)
        # Get reward of agent 0 --> assume cooperative rewards/ same reward for all agents
        rewards = batch.rewards[0]
        dones = batch.dones[1:]
        filled = batch.filled
        action_masks = batch.action_mask

        if self.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)

        # (n_agents, ep_length, batch_size, n_actions)
        q_values, _ = self.critic(obss, hiddens=None)
        q_values = torch.stack(q_values)
        # sum over all agents for cooperative VDN estimate
        chosen_q_values = self.mixer(
            q_values[:, :-1].gather(-1, actions).squeeze(-1),
            torch.concat(list(obss[:, :-1]), dim=-1),
        )

        # compute target
        with torch.no_grad():
            target_q_values, _ = self.target(obss, hiddens=None)
            target_q_values = torch.stack(target_q_values)[:, 1:]
            if action_masks is not None:
                target_q_values[action_masks[:, 1:] == 0] = -1e8

            if self.double_q:
                q_values_clone = q_values.clone().detach()[:, 1:]
                if action_masks is not None:
                    q_values_clone[action_masks[:, 1:] == 0] = -1e8
                a_prime = q_values_clone.argmax(-1)
                target_qs = target_q_values.gather(-1, a_prime.unsqueeze(-1)).squeeze(
                    -1
                )
            else:
                target_qs, _ = target_q_values.max(dim=-1)

            target_qs = self.target_mixer(
                target_qs,
                torch.concat(list(obss[:, 1:]), dim=-1),
            )
        returns = rewards + self.gamma * target_qs.detach() * (1 - dones)

        if self.standardise_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)

        loss = torch.nn.functional.mse_loss(
            chosen_q_values, returns.detach(), reduction="none"
        )
        return (loss * filled).sum() / filled.sum()

    def soft_update(self, t):
        super().soft_update(t)
        try:
            source, target = self.mixer, self.target_mixer
        except AttributeError:  # fix for when qmix has not initialised a mixer yet
            return
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    def hard_update(self):
        super().hard_update()
        try:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        except AttributeError:  # fix for when qmix has not initialised a mixer yet
            return
