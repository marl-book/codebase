import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from gym.spaces import flatdim
from blazingma.utils.models import MultiAgentFCNetwork

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
    def __init__(self, obs_space, action_space, cfg):
        super(Policy, self).__init__()

        self.n_agents = len(obs_space)
        obs_shape = [flatdim(o) for o in obs_space]
        state_shape = [flatdim(obs_space) for _ in obs_space]
        action_shape = [flatdim(a) for a in action_space]

        self.actor = MultiAgentFCNetwork(
            obs_shape, list(cfg.model.actor.layers), action_shape
        )
        for layers in self.actor.models:
            nn.init.orthogonal_(layers[-1].weight.data, gain=0.01)
        self.gate = nn.Parameter(torch.rand(1, 1))

        self.vf = MultiAgentFCNetwork(state_shape, list(cfg.model.critic.layers), len(action_shape)*[1])
        self.target_vf = MultiAgentFCNetwork(state_shape, list(cfg.model.critic.layers), len(action_shape)*[1])
        self.soft_update(1.0)

        # the critic takes the state + actions of others
        input_shape = []
        for act in action_space:
            input_shape += [sum(obs_shape) + flatdim(action_space) - flatdim(act)]
        self.critic = MultiAgentFCNetwork(input_shape, list(cfg.model.critic.layers), action_shape)

        input_shape = []

        for i in range(self.n_agents):
            input_shape += [np.prod([a.n for j, a in enumerate(action_space) if i != j])]
        self.mixing = MultiAgentFCNetwork(input_shape, [64, 64], self.n_agents * [1])
        
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
        return torch.cat(self.vf(inputs), dim=-1)

    def get_target_value(self, inputs):
        return torch.cat(self.target_vf(inputs), dim=-1)

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
        source, target = self.vf, self.target_vf
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)