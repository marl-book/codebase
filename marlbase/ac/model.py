from collections import defaultdict

from einops import rearrange
from gymnasium.spaces import flatdim
import torch
from torch.distributions import Categorical
import torch.nn as nn
from torch import optim

from marlbase.utils.models import MultiAgentIndependentNetwork, MultiAgentSharedNetwork
from marlbase.utils.utils import MultiCategorical, compute_nstep_returns
from marlbase.utils.standardise_stream import RunningMeanStd


def _split_batch(splits):
    def thunk(batch):
        return torch.split(batch, splits, dim=-1)

    return thunk


class A2CNetwork(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        actor,
        critic,
        device,
    ):
        super(A2CNetwork, self).__init__()
        self.gamma = cfg.gamma
        self.entropy_coef = cfg.entropy_coef
        self.n_steps = cfg.n_steps
        self.grad_clip = cfg.grad_clip
        self.value_loss_coef = cfg.value_loss_coef
        self.device = device

        self.n_agents = len(obs_space)
        obs_dims = [flatdim(o) for o in obs_space]
        act_dims = [flatdim(a) for a in action_space]

        if not actor.parameter_sharing:
            self.actor = MultiAgentIndependentNetwork(
                obs_dims,
                list(actor.layers),
                act_dims,
                actor.use_rnn,
                actor.use_orthogonal_init,
            )
        else:
            self.actor = MultiAgentSharedNetwork(
                obs_dims,
                list(actor.layers),
                act_dims,
                actor.parameter_sharing,
                actor.use_rnn,
                actor.use_orthogonal_init,
            )

        self.centralised_critic = critic.centralised
        critic_obs_shape = (
            self.n_agents * [sum(obs_dims)] if critic.centralised else obs_dims
        )

        if not critic.parameter_sharing:
            self.critic = MultiAgentIndependentNetwork(
                critic_obs_shape,
                list(critic.layers),
                [1] * self.n_agents,
                critic.use_rnn,
                critic.use_orthogonal_init,
            )
            self.target_critic = MultiAgentIndependentNetwork(
                critic_obs_shape,
                list(critic.layers),
                [1] * self.n_agents,
                critic.use_rnn,
                critic.use_orthogonal_init,
            )
        else:
            self.critic = MultiAgentSharedNetwork(
                critic_obs_shape,
                list(critic.layers),
                [1] * self.n_agents,
                critic.parameter_sharing,
                critic.use_rnn,
                critic.use_orthogonal_init,
            )
            self.target_critic = MultiAgentSharedNetwork(
                critic_obs_shape,
                list(critic.layers),
                [1] * self.n_agents,
                critic.parameter_sharing,
                critic.use_rnn,
                critic.use_orthogonal_init,
            )

        self.soft_update(1.0)
        self.to(device)

        optimizer = getattr(optim, cfg.optimizer)
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer_class = optimizer

        lr = cfg.lr
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        self.standardise_returns = cfg.standardise_returns
        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)

        self.split_obs = _split_batch([flatdim(s) for s in obs_space])
        self.split_act = _split_batch(self.n_agents * [1])

        print(self)

    def init_critic_hiddens(self, batch_size, target=False):
        if target:
            return self.target_critic.init_hiddens(batch_size, self.device)
        else:
            return self.critic.init_hiddens(batch_size, self.device)

    def init_actor_hiddens(self, batch_size):
        return self.actor.init_hiddens(batch_size, self.device)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError(
            "Forward not implemented. Use act, get_value, get_target_value or evaluate_actions instead."
        )

    def get_dist(self, action_logits, action_mask=None):
        if action_mask is not None:
            masked_logits = []
            for logits, mask in zip(action_logits, action_mask):
                masked_logits.append(logits * mask + (1 - mask) * -1e8)
            action_logits = masked_logits

        dist = MultiCategorical(
            [Categorical(logits=logits) for logits in action_logits]
        )
        return dist

    def act(self, inputs, actor_hiddens, action_mask=None):
        inputs = [i.unsqueeze(0) for i in inputs]
        actor_logits, actor_hiddens = self.actor(inputs, actor_hiddens)
        actor_logits = [logits.squeeze(0) for logits in actor_logits]
        dist = self.get_dist(actor_logits, action_mask)
        actions = dist.sample()
        return torch.stack(actions, dim=0), actor_hiddens

    def get_value(self, inputs, critic_hiddens, target=False):
        if self.centralised_critic:
            inputs = self.n_agents * [torch.cat(inputs, dim=-1)]

        if target:
            values, critic_hiddens = self.target_critic(inputs, critic_hiddens)
        else:
            values, critic_hiddens = self.critic(inputs, critic_hiddens)
        return torch.cat(values, dim=-1), critic_hiddens

    def evaluate_actions(
        self,
        inputs,
        action,
        critic_hiddens,
        actor_hiddens,
        action_mask=None,
        state=None,
    ):
        if state is None:
            state = inputs
        value, critic_hiddens = self.get_value(state, critic_hiddens)
        actor_features, actor_hiddens = self.actor(inputs, actor_hiddens)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = torch.stack(dist.entropy(), dim=-1).sum(dim=-1)

        return (value, action_log_probs, dist_entropy, critic_hiddens, actor_hiddens)

    def soft_update(self, t):
        source, target = self.critic, self.target_critic
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    def update(self, batch, step):
        with torch.no_grad():
            next_value, _ = self.get_value(
                self.split_obs(batch.obss), critic_hiddens=None, target=True
            )

        if self.standardise_returns:
            next_value = next_value * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean

        batch_done = batch.dones.float().unsqueeze(-1).repeat(1, 1, self.n_agents)
        returns = compute_nstep_returns(
            batch.rewards, batch_done, next_value, self.n_steps, self.gamma
        )
        if self.standardise_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)

        values, action_log_probs, entropy, _, _ = self.evaluate_actions(
            self.split_obs(batch.obss[:-1]),
            self.split_act(batch.actions),
            critic_hiddens=None,
            actor_hiddens=None,
            action_mask=rearrange(batch.action_masks[:-1], "E B N A -> N E B A")
            if batch.action_masks is not None
            else None,
        )

        advantage = returns - values

        actor_loss = (
            -(action_log_probs * advantage.detach()).sum(dim=-1)
            - self.entropy_coef * entropy
        )
        actor_loss = (actor_loss * batch.filled).sum() / batch.filled.sum()
        value_loss = (returns - values).pow(2).sum(dim=-1)
        value_loss = (value_loss * batch.filled).sum() / batch.filled.sum()

        loss = actor_loss + self.value_loss_coef * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        if (
            self.target_update_interval_or_tau > 1.0
            and step % self.target_update_interval_or_tau == 0
        ):
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": ((entropy * batch.filled).sum() / batch.filled.sum()).item(),
        }


class PPONetwork(A2CNetwork):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        actor,
        critic,
        device,
    ):
        super(PPONetwork, self).__init__(
            obs_space, action_space, cfg, actor, critic, device
        )
        self.num_epochs = cfg.num_epochs
        self.ppo_clip = cfg.ppo_clip

    def update(self, batch, step):
        # compute returns
        with torch.no_grad():
            next_value, _ = self.get_value(
                self.split_obs(batch.obss), critic_hiddens=None, target=True
            )

        if self.standardise_returns:
            next_value = next_value * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean

        batch_done = batch.dones.float().unsqueeze(-1).repeat(1, 1, self.n_agents)
        returns = compute_nstep_returns(
            batch.rewards, batch_done, next_value, self.n_steps, self.gamma
        ).detach()
        if self.standardise_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)

        # compute old policy log probs
        with torch.no_grad():
            actor_features, _ = self.actor(self.split_obs(batch.obss[:-1]), None)
            dist = self.get_dist(
                actor_features,
                action_mask=rearrange(batch.action_masks[:-1], "E B N A -> N E B A")
                if batch.action_masks is not None
                else None,
            )
            old_action_log_probs = torch.cat(
                dist.log_probs(self.split_act(batch.actions)), dim=-1
            ).detach()

        metrics = defaultdict(list)
        for _ in range(self.num_epochs):
            # sample from current policy
            values, action_log_probs, entropy, _, _ = self.evaluate_actions(
                self.split_obs(batch.obss[:-1]),
                self.split_act(batch.actions),
                critic_hiddens=None,
                actor_hiddens=None,
                action_mask=rearrange(batch.action_masks[:-1], "E B N A -> N E B A")
                if batch.action_masks is not None
                else None,
            )

            # compute advantage and value loss
            advantage = returns - values
            value_loss = advantage.pow(2).sum(dim=-1)

            # compute policy loss
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantage.detach()
            surr2 = (
                torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
                * advantage.detach()
            )
            actor_loss = (
                -torch.min(surr1, surr2).sum(dim=-1) - self.entropy_coef * entropy
            )

            # apply masks and compute total loss per epoch
            actor_loss = (actor_loss * batch.filled).sum() / batch.filled.sum()
            value_loss = (value_loss * batch.filled).sum() / batch.filled.sum()
            loss = actor_loss + self.value_loss_coef * value_loss

            # optimisation step
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()

            metrics["loss"].append(loss.item())
            metrics["actor_loss"].append(actor_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(
                ((entropy * batch.filled).sum() / batch.filled.sum()).item()
            )

        # update target network after last epoch
        if (
            self.target_update_interval_or_tau > 1.0
            and step % self.target_update_interval_or_tau == 0
        ):
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

        return {key: sum(values) / len(values) for key, values in metrics.items()}
