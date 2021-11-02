import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from gym.spaces import flatdim
from blazingma.utils.models import MultiAgentFCNetwork, MultiAgentSEPSNetwork
from typing import List


class MultiCategorical:
    def __init__(self, categoricals):
        self.categoricals = categoricals

    def __getitem__(self, key):
        return self.categoricals[key]

    def sample(self):
        return [c.sample().unsqueeze(-1) for c in self.categoricals]

    def log_probs(self, actions):

        return [
            c.log_prob(a.squeeze(-1)).unsqueeze(-1)
            for c, a in zip(self.categoricals, actions)
        ]

    def mode(self):
        return [c.mode for c in self.categoricals]

    def entropy(self):
        return [c.entropy() for c in self.categoricals]


class Policy(nn.Module):
    def __init__(self, obs_space, action_space, actor, critic, device):
        super(Policy, self).__init__()

        self.n_agents = len(obs_space)
        obs_shape = [flatdim(o) for o in obs_space]
        action_shape = [flatdim(a) for a in action_space]

        if not actor.parameter_sharing:
            self.actor = MultiAgentFCNetwork(
                obs_shape, list(actor.layers), action_shape
            )
        else:
            self.actor = MultiAgentSEPSNetwork(
                obs_shape, list(actor.layers) + [action_shape[0]], actor.parameter_sharing
            )

        for layers in self.actor.independent:
            nn.init.orthogonal_(layers[-1].weight.data, gain=0.01)

        self.centralised_critic = critic.centralised
        critic_obs_shape = self.n_agents * [sum(obs_shape)] if critic.centralised else obs_shape 

        if not critic.parameter_sharing:
            self.critic = MultiAgentFCNetwork(critic_obs_shape, list(critic.layers), len(action_shape)*[1])
            self.target_critic = MultiAgentFCNetwork(critic_obs_shape, list(critic.layers), len(action_shape)*[1])
        else:
            self.critic = MultiAgentSEPSNetwork(critic_obs_shape, list(critic.layers) + [1], critic.parameter_sharing)
            self.target_critic = MultiAgentSEPSNetwork(critic_obs_shape, list(critic.layers) + [1], critic.parameter_sharing)

        self.soft_update(1.0)
        self.to(device)

        print(self)
        
    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_dist(self, actor_features, action_mask):
        if action_mask:
            action_mask = [-9999999 * (1 - action_mask) for a in action_mask]
        else:
            action_mask = len(actor_features) * [0]

        dist = MultiCategorical(
            [Categorical(logits=x + s) for x, s in zip(actor_features, action_mask)]
        )
        return dist

    def act(self, inputs, action_mask=None):
        actor_features = self.actor(inputs)
        dist = self.get_dist(actor_features, action_mask)
        action = dist.sample()
        return action

    def get_value(self, inputs):
        if self.centralised_critic:
            inputs = self.n_agents * [torch.cat(inputs, dim=-1)]

        return torch.cat(self.critic(inputs), dim=-1)

    def get_target_value(self, inputs):
        if self.centralised_critic:
            inputs = self.n_agents * [torch.cat(inputs, dim=-1)]
            
        return torch.cat(self.target_critic(inputs), dim=-1)

    def evaluate_actions(self, inputs, action, action_mask=None, state=None):
        if not state:
            state = inputs
        value = self.get_value(state)
        actor_features = self.actor(inputs)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = dist.entropy()
        dist_entropy = sum([d.mean() for d in dist_entropy])

        return (
            value,
            action_log_probs,
            dist_entropy,
        )

    def soft_update(self, t):
        source, target = self.critic, self.target_critic
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)
