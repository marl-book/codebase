from itertools import product
import random

from gym.spaces import flatdim
import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn as nn

from fastmarl.utils.agent_models import PolicyModel
from fastmarl.utils.models import MultiAgentSEPSNetwork, MultiAgentFCNetwork
from fastmarl.utils.standarize_stream import RunningMeanStd
from fastmarl.utils.utils import MultiCategorical, to_onehot


class JointQNetwork(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        layers,
        critic,
        agent_model,
        sample_action_value,
        n_samples,
        device,
    ):
        super().__init__()
        hidden_size = list(layers)
        gamma = cfg.gamma
        lr = cfg.lr
        grad_clip = cfg.grad_clip
        optimizer = getattr(optim, cfg.optimizer)

        self.action_space = action_space

        self.n_agents = len(obs_space)
        self.obs_shape = [flatdim(o) for o in obs_space]
        self.action_shape = [flatdim(a) for a in action_space]

        # input for agent i's network is the observation of agent i and the action of all other agents
        input_sizes = [
            self.obs_shape[i] + sum([self.action_shape[j] for j in range(self.n_agents) if j != i])
            for i in range(self.n_agents)
        ]

        # MultiAgentFCNetwork is much faster that MultiAgentSepsNetwork
        # We would like to keep this, so a simple `if` switch is implemented below
        if not critic.parameter_sharing:
            self.target = MultiAgentFCNetwork(input_sizes, hidden_size, self.action_shape, critic.use_orthogonal_init)
            self.target = MultiAgentFCNetwork(input_sizes, hidden_size, self.action_shape, critic.use_orthogonal_init)
        else:
            self.critic = MultiAgentSEPSNetwork(
                input_sizes, hidden_size, self.action_shape, critic.parameter_sharing, critic.use_orthogonal_init,
            )
            self.target = MultiAgentSEPSNetwork(
                input_sizes, hidden_size, self.action_shape, critic.parameter_sharing, critic.use_orthogonal_init,
            )
        
        # Initialise agent models for each agent
        self.agent_models = nn.ModuleList(
            [
                PolicyModel(
                    self.obs_shape[i],
                    agent_model.base_layers,
                    agent_model.policy_layers,
                    [act_dim for j, act_dim in enumerate(self.action_shape) if j != i],
                    use_orthogonal_init=agent_model.use_orthogonal_init,
                ) for i in range(self.n_agents)
            ]
        )
        self.sample_action_value = sample_action_value
        self.n_samples = n_samples

        self.soft_update(1.0)

        for param in self.target.parameters():
            param.requires_grad = False

        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer_class = optimizer

        self.optimizer = optimizer(list(self.critic.parameters()) + list(self.agent_models.parameters()), lr=lr)

        self.gamma = gamma
        self.grad_clip = grad_clip
        self.device = device

        self.updates = 0
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        self.standardize_returns = cfg.standardize_returns
        self.ret_ms = RunningMeanStd(shape=(self.n_agents,))

        print(self)

    def forward(self, inputs):
        raise NotImplemented
    
    def predict_agent_actions(self, inputs, sample=True):
        other_action_logits = torch.stack([model(x) for model, x in zip(self.agent_models, inputs)], dim=0)
        if sample:
            other_action_dists = [
                MultiCategorical(
                    [Categorical(logits=act_logits) for act_logits in other_action_logits[i]]
                ) for i in range(self.n_agents)
            ]
            other_actions = torch.stack([torch.stack(dist.sample()) for dist in other_action_dists], dim=0).squeeze(-1)
        else:
            _, other_actions = other_action_logits.max(dim=-1)
        return other_action_logits, other_actions

    def build_critic_inputs(self, inputs, other_actions=None, sample=True):
        if other_actions is None:
            _, other_actions = self.predict_agent_actions(inputs, sample)
        # print(f"Build critic inputs -- Inputs shape: {[model_input.shape for model_input in inputs]}")
        # print(f"Build critic inputs -- Other actions shape: {[other_act.shape for other_act in other_actions]}")
        other_actions_onehot = [
            torch.concat([to_onehot(other_act, self.action_shape[j]) for j, other_act in enumerate(other_actions[i])], dim=-1)
            for i in range(self.n_agents)
        ]
        # print(f"Build critic inputs -- Other actions onehot shape: {[other_onehot.shape for other_onehot in other_actions_onehot]}")
        critic_inputs = [torch.cat([model_input, other_act_onehot], dim=-1) for model_input, other_act_onehot in zip(inputs, other_actions_onehot)]
        return critic_inputs

    def _get_all_other_actions(self):
        # n_agents lists of (n_agents - 1, batchsize) where batchsize is the number of possible joint actions of all other agents
        # for all agents
        other_actions = []
        for i in range(self.n_agents):
            other_action_combinations = list(product(*[torch.arange(self.action_shape[j], device=self.device) for j in range(self.n_agents) if j != i]))
            other_acts_ints = torch.stack([torch.tensor(other_action_combination, device=self.device) for other_action_combination in other_action_combinations], dim=0)
            other_actions.append(other_acts_ints.swapaxes(0, 1))
        return other_actions
    
    def compute_action_values(self, model, inputs):
        if self.sample_action_value:
            # compute n_samples Q-values with sampled actions of other agents and compute
            # average over these Q-values
            inputs = [x.repeat((self.n_samples,) + tuple([1] * x.dim())) for x in inputs]
            # print(f"Compute action values -- Inputs shape: {[model_input.shape for model_input in inputs]}")
            critic_inputs = self.build_critic_inputs(inputs, other_actions=None, sample=True)
            action_values = torch.stack(model(critic_inputs), dim=0)
            action_values = action_values.mean(dim=1)
        else:
            # compute expected Q-values under models of other agents (i.e. average over all possible actions)
            other_actions = self._get_all_other_actions()
            if inputs[0].dim() == 1:
                inputs = [x.unsqueeze(0) for x in inputs]
            batchsize = inputs[0].shape[0]
            # print(f"Compute action values -- Inputs shape: {[model_input.shape for model_input in inputs]}")
            # print(f"Compute action values -- Other actions shape: {[other_act.shape for other_act in other_actions]}")
            num_action_combinations = [other_acts.shape[-1] for other_acts in other_actions]
            inputs = [x.unsqueeze(0).repeat((num_action_combinations[i],) + tuple([1] * x.dim())) for i, x in enumerate(inputs)]
            other_actions = [other_acts.unsqueeze(-1).repeat((1, 1, batchsize,)) for other_acts in other_actions]
            critic_inputs = self.build_critic_inputs(inputs, other_actions=other_actions)
            action_values = torch.stack(model(critic_inputs), dim=0)
            action_values = action_values.mean(dim=1)
            if action_values.shape[1] == 1:
                action_values = action_values.squeeze(1)
            # print(f"Compute action values -- Action values shape: {action_values.shape}")
        return action_values

    def act(self, inputs, epsilon):
        if epsilon > random.random():
            actions = self.action_space.sample()
        else:
            with torch.no_grad():
                inputs = [torch.from_numpy(x).to(self.device) for x in inputs]
                action_values = self.compute_action_values(self.critic, inputs)
                actions = [av.argmax(dim=0).cpu().item() for av in action_values]
        return actions

    def update(self, batch):
        obs = [batch[f"obs{i}"] for i in range(self.n_agents)]
        nobs = [batch[f"next_obs{i}"] for i in range(self.n_agents)]
        action = torch.stack([batch[f"act{i}"].long() for i in range(self.n_agents)])
        rewards = torch.stack(
            [batch["rew"][:, i].view(-1, 1) for i in range(self.n_agents)]
        )
        done = batch["done"]

        # Use true actions of other agents where available and otherwise use predicted actions
        other_actions = torch.stack(
            [
                torch.stack([action[j] for j in range(self.n_agents) if j != i], dim=0)
                for i in range(self.n_agents)
            ],
            dim=0,
        ).squeeze(-1)
        critic_inputs = self.build_critic_inputs(obs, other_actions)

        # Compute critic loss
        with torch.no_grad():
            q_tp1_values = self.compute_action_values(self.critic, nobs)
            q_next_states = self.compute_action_values(self.target, nobs)
        all_q_states = torch.stack(self.critic(critic_inputs))

        _, a_prime = q_tp1_values.max(-1)
        target_next_states = q_next_states.gather(2, a_prime.unsqueeze(-1))
        target_states = rewards + self.gamma * target_next_states * (1 - done)

        if self.standardize_returns:
            self.ret_ms.update(target_states)
            target_states = (
                target_states - self.ret_ms.mean.view(-1, 1, 1)
            ) / torch.sqrt(self.ret_ms.var.view(-1, 1, 1))

        q_states = all_q_states.gather(2, action)

        critic_loss = torch.nn.functional.mse_loss(q_states, target_states)

        # Compute agent model loss
        other_actions_logits, _ = self.predict_agent_actions(obs)
        agent_model_loss = sum([
            torch.nn.functional.cross_entropy(
                other_actions_logits[i].view(-1, flatdim(self.action_space[0])),
                other_actions[i].view(-1),
                reduction="mean",
            ) for i in range(self.n_agents)
        ])

        loss = critic_loss + agent_model_loss

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.agent_models.parameters(), self.grad_clip)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_from_target()

    def update_from_target(self):
        if (
            self.target_update_interval_or_tau > 1.0
            and self.updates % self.target_update_interval_or_tau == 0
        ):
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)
        self.updates += 1

    def soft_update(self, t):
        source, target = self.critic, self.target
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)
