import random

import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import flatdim
from blazingma.utils.models import MultiAgentSEPSNetwork, MultiAgentFCNetwork


class QNetwork(nn.Module):
    def __init__(
        self, obs_space, action_space, cfg, layers, critic, device,
    ):
        super().__init__()
        hidden_size = list(layers)
        gamma = cfg.gamma
        lr = cfg.lr
        grad_clip = cfg.grad_clip
        optimizer = getattr(optim, cfg.optimizer)
        optim_eps = cfg.optim_eps

        self.action_space = action_space

        self.n_agents = len(obs_space)
        obs_shape = [flatdim(o) for o in obs_space]
        action_shape = [flatdim(a) for a in action_space]

        # MultiAgentFCNetwork is much faster that MultiAgentSepsNetwork
        # We would like to keep this, so a simple `if` switch is implemented below
        if not critic.parameter_sharing:
            self.critic = MultiAgentFCNetwork(obs_shape, hidden_size, action_shape)
            self.target = MultiAgentFCNetwork(obs_shape, hidden_size, action_shape)
        else:
            self.critic = MultiAgentSEPSNetwork(obs_shape, hidden_size + [action_shape[0]], critic.parameter_sharing)
            self.target = MultiAgentSEPSNetwork(obs_shape, hidden_size + [action_shape[0]], critic.parameter_sharing)

        self.soft_update(1.0)

        for param in self.target.parameters():
            param.requires_grad = False

        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)

        self.optimizer = optimizer(self.critic.parameters(), lr=lr, eps=optim_eps)

        self.gamma = gamma
        self.grad_clip = grad_clip
        self.device = device

        self.updates = 0
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        print(self)

    def forward(self, inputs):
        raise NotImplemented

    def act(self, inputs, epsilon):
        if epsilon > random.random():
            actions = self.action_space.sample()
            return actions
        with torch.no_grad():
            inputs = [torch.from_numpy(i).to(self.device) for i in inputs]
            actions = [x.argmax(dim=0).cpu().item() for x in self.critic(inputs)]

        return actions

    def update(self, batch):

        obs = [batch[f"obs{i}"] for i in range(self.n_agents)]
        nobs = [batch[f"next_obs{i}"] for i in range(self.n_agents)]
        action = [batch[f"act{i}"].long() for i in range(self.n_agents)]
        rewards = [batch["rew"][:, i].view(-1, 1) for i in range(self.n_agents)]
        done = batch["done"]

        with torch.no_grad():
            q_tp1_values = self.critic(nobs)
            q_next_states = self.target(nobs)
        all_q_states = self.critic(obs)

        loss = 0.0

        for i in range(self.n_agents):
            _, a_prime = q_tp1_values[i].max(1)
            target_next_states = q_next_states[i].gather(1, a_prime.unsqueeze(1))
            target_states = rewards[i] + self.gamma * target_next_states * (1 - done)
            q_states = all_q_states[i].gather(1, action[i])
            loss += torch.nn.functional.mse_loss(q_states, target_states)

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_from_target()

    def update_from_target(self):
        if self.target_update_interval_or_tau > 1.0 and self.updates % self.target_update_interval_or_tau == 0:
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)
        self.updates += 1

    def soft_update(self, t):
        source, target = self.critic, self.target
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


class VDNetwork(QNetwork):
    def update(self, batch):
        
        obs = [batch[f"obs{i}"] for i in range(self.n_agents)]
        nobs = [batch[f"next_obs{i}"] for i in range(self.n_agents)]
        action = [batch[f"act{i}"].long() for i in range(self.n_agents)]
        rewards = batch["rew"].view(-1, 1)
        done = batch["done"]

        with torch.no_grad():
            q_tp1_values = self.critic(nobs)
            q_next_states = self.target(nobs)
        all_q_states = self.critic(obs)

        target_states = 0
        q_states = 0

        for i in range(self.n_agents):
            _, a_prime = q_tp1_values[i].max(1)
            target_next_states = q_next_states[i].gather(1, a_prime.unsqueeze(1))
            target_states += self.gamma * target_next_states * (1 - done)
            q_states += all_q_states[i].gather(1, action[i])

        target_states += rewards
        loss = torch.nn.functional.mse_loss(q_states, target_states)

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_from_target()


