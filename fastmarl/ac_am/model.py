from gym.spaces import flatdim
import torch
from torch.distributions import Categorical
import torch.nn as nn
from torch import optim

from fastmarl.utils.agent_models import EncoderDecoderModel
from fastmarl.utils.models import MultiAgentFCNetwork, MultiAgentSEPSNetwork
from fastmarl.utils.utils import MultiCategorical
from fastmarl.utils.standarize_stream import RunningMeanStd


def _split_batch(splits):
    def thunk(batch):
        return torch.split(batch, splits, dim=-1)

    return thunk

@torch.jit.script
def compute_returns(rewards, done, next_value, gamma: float):
    returns = [next_value]
    for i in range(len(rewards) - 1, -1, -1):
        ret = rewards[i] + gamma * returns[0] * (1 - done[i, :].unsqueeze(1))
        returns.insert(0, ret)
    return returns


class ActorCriticAgentModel(nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            cfg,
            actor,
            critic,
            agent_model,
            device,
        ):
        super(ActorCriticAgentModel, self).__init__()
        self.gamma = cfg.gamma
        self.entropy_coef = cfg.entropy_coef
        self.n_steps = cfg.n_steps
        self.grad_clip = cfg.grad_clip
        self.value_loss_coef = cfg.value_loss_coef

        self.n_agents = len(obs_space)
        self.obs_shape = [flatdim(o) for o in obs_space]
        self.action_shape = [flatdim(a) for a in action_space]

        # initialise encoder-decoder models
        assert agent_model.decoder.decode_observations or agent_model.decoder.decode_actions, "Decoder must decode either actions or observations or both"
        if agent_model.decoder.decode_actions:
            self.decode_actions = True
            output_actions = [
                [act_shape for j, act_shape in enumerate(self.action_shape) if j != i]
                for i in range(self.n_agents)
            ]
        else:
            self.decode_actions = False
            output_actions = None
        if agent_model.decoder.decode_observations:
            self.decode_observations = True
            output_observations = [
                [obs_dim for j, obs_dim in enumerate(self.obs_shape) if j != i]
                for i in range(self.n_agents)
            ]
        else:
            self.decode_observations = False
            output_observations = None
        self.encoder_decoder_models = nn.ModuleList(
            [
                EncoderDecoderModel(
                    self.obs_shape[i],
                    agent_model.encoder.layers,
                    agent_model.latent_dim,
                    agent_model.decoder.base_layers,
                    agent_model.decoder.head_layers,
                    observation_output_sizes=output_observations[i] if output_observations is not None else None,
                    action_output_sizes=output_actions[i] if output_actions is not None else None,
                    use_orthogonal_init=agent_model.use_orthogonal_init,
                ) for i in range(self.n_agents)
            ]
        )

        if not actor.parameter_sharing:
            self.actor = MultiAgentFCNetwork(
                [obs_dim + agent_model.latent_dim for obs_dim in self.obs_shape],
                list(actor.layers),
                self.action_shape,
                actor.use_orthogonal_init,
            )
        else:
            self.actor = MultiAgentSEPSNetwork(
                [obs_dim + agent_model.latent_dim for obs_dim in self.obs_shape],
                list(actor.layers),
                self.action_shape,
                actor.parameter_sharing,
                actor.use_orthogonal_init,
            )

        if actor.use_orthogonal_init:
            for layers in self.actor.independent:
                nn.init.orthogonal_(layers[-1].weight.data, gain=0.01)

        self.centralised_critic = critic.centralised
        if critic.centralised:
            critic_obs_shape = self.n_agents * [sum(self.obs_shape) + agent_model.latent_dim]
        else:
            critic_obs_shape = [obs_dim + agent_model.latent_dim for obs_dim in self.obs_shape]

        if not critic.parameter_sharing:
            self.critic = MultiAgentFCNetwork(critic_obs_shape, list(critic.layers), [1] * self.n_agents, critic.use_orthogonal_init)
            self.target_critic = MultiAgentFCNetwork(critic_obs_shape, list(critic.layers), [1] * self.n_agents, critic.use_orthogonal_init)
        else:
            self.critic = MultiAgentSEPSNetwork(critic_obs_shape, list(critic.layers), [1] * self.n_agents, critic.parameter_sharing, critic.use_orthogonal_init)
            self.target_critic = MultiAgentSEPSNetwork(critic_obs_shape, list(critic.layers), [1] * self.n_agents, critic.parameter_sharing, critic.use_orthogonal_init)

        self.soft_update(1.0)
        self.to(device)

        optimizer = getattr(optim, cfg.optimizer)
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer_class = optimizer

        lr = cfg.lr
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        self.standardize_returns = cfg.standardize_returns
        self.ret_ms = RunningMeanStd(shape=(self.n_agents,))

        self.split_obs = _split_batch([flatdim(s) for s in obs_space])
        self.split_act = _split_batch(self.n_agents * [1])

        print(self)
        
    def forward(self, inputs, masks):
        raise NotImplementedError("Forward not implemented. Use act, get_value, get_target_value or evaluate_actions instead.")
    
    def get_dist(self, actor_features, action_mask):
        if action_mask:
            action_mask = [-9999999 * (1 - action_mask) for a in action_mask]
        else:
            action_mask = len(actor_features) * [0]

        dist = MultiCategorical(
            [Categorical(logits=x + s) for x, s in zip(actor_features, action_mask)]
        )
        return dist
    
    def _add_latents_to_inputs(self, inputs, embeddings=None):
        if embeddings is None:
            embeddings = [self.encoder_decoder_models[i].encode(inputs[i]) for i in range(self.n_agents)]
        return [torch.cat([inputs[i], embeddings[i].detach()], dim=-1) for i in range(self.n_agents)]

    def act(self, inputs, action_mask=None):
        actor_inputs = self._add_latents_to_inputs(inputs)
        actor_features = self.actor(actor_inputs)
        dist = self.get_dist(actor_features, action_mask)
        action = dist.sample()
        return action

    def get_value(self, inputs, embeddings=None):
        if embeddings is None:
            embeddings = [self.encoder_decoder_models[i].encode(inputs[i]) for i in range(self.n_agents)]
        if self.centralised_critic:
            inputs = self.n_agents * [torch.cat(inputs, dim=-1)]
        inputs = self._add_latents_to_inputs(inputs, embeddings)
        return torch.cat(self.critic(inputs), dim=-1)

    def get_target_value(self, inputs, embeddings=None):
        if embeddings is None:
            embeddings = [self.encoder_decoder_models[i].encode(inputs[i]) for i in range(self.n_agents)]
        if self.centralised_critic:
            inputs = self.n_agents * [torch.cat(inputs, dim=-1)]
        inputs = self._add_latents_to_inputs(inputs, embeddings)
        return torch.cat(self.target_critic(inputs), dim=-1)

    def evaluate_actions(self, inputs, action, embeddings=None, action_mask=None, state=None):
        if not state:
            state = inputs
        value = self.get_value(state, embeddings)

        actor_inputs = self._add_latents_to_inputs(inputs, embeddings)
        actor_features = self.actor(actor_inputs)
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
    
    def update(self, batch_obs, batch_act, batch_rew, batch_done, step):
        current_obs = self.split_obs(batch_obs[:-1])
        acts = self.split_act(batch_act)
        next_obs = self.split_obs(batch_obs[self.n_steps, :, :])

        with torch.no_grad():
            next_value = self.get_target_value(next_obs)

        if self.standardize_returns:
            next_value = next_value * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean
        returns = compute_returns(batch_rew, batch_done, next_value, self.gamma)

        encoder_decoder_outs = [
            encoder_decoder_model(current_obs_i) for encoder_decoder_model, current_obs_i in zip(self.encoder_decoder_models, current_obs)
        ]
        embeddings = [out[0] for out in encoder_decoder_outs]
        decoder_observations = [out[1][0] for out in encoder_decoder_outs]
        decoder_actions = [out[1][1] for out in encoder_decoder_outs]

        values, action_log_probs, entropy = self.evaluate_actions(current_obs, acts, embeddings)

        returns = torch.stack(returns)[:-1]

        if self.standardize_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)

        advantage = returns - values

        actor_loss = (
            -(action_log_probs * advantage.detach()).sum(dim=2).mean()
            - self.entropy_coef * entropy
        )
        value_loss = (returns - values).pow(2).sum(dim=2).mean()

        decode_obs_targets = self.split_obs(batch_obs[1:, :, :])

        if self.decode_observations:
            observation_decode_loss = torch.sum(
                torch.stack([
                    nn.functional.mse_loss(
                        decode_obs,
                        torch.stack([decode_obs_targets[j] for j in range(self.n_agents) if j != i], dim=0),
                        reduction="mean"
                    ) for i, decode_obs in enumerate(decoder_observations)
                ])
            )
        else:
            observation_decode_loss = 0

        action_decode_loss = 0
        if self.decode_actions:
            for i, decode_acts in enumerate(decoder_actions):
                decode_index = 0
                for j in range(self.n_agents):
                    if j == i:
                        continue
                    decode_action = decode_acts[decode_index].view(-1, self.action_shape[j])
                    action_target = acts[j].view(-1).long()
                    action_decode_loss += nn.functional.cross_entropy(decode_action, action_target, reduction="mean")
                    decode_index += 1
        decoder_loss = observation_decode_loss + action_decode_loss

        loss = actor_loss + self.value_loss_coef * value_loss + decoder_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.target_update_interval_or_tau > 1.0 and step % self.target_update_interval_or_tau == 0:
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)
