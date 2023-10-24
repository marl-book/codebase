import torch
import torch.nn as nn

from fastmarl.utils.models import make_fc


class PolicyModel(nn.Module):
    def __init__(self, input_size, base_layers, policy_layers, output_sizes, use_orthogonal_init=True):
        super().__init__()

        self.base_model = make_fc([input_size] + list(base_layers), final_activation=nn.ReLU, use_orthogonal_init=use_orthogonal_init)
        self.base_dim = base_layers[-1]
        self.policy_heads = nn.ModuleList(
            [make_fc([self.base_dim] + list(policy_layers) + [output_size], use_orthogonal_init=use_orthogonal_init) for output_size in output_sizes]
        )
    
    def forward(self, x):
        x = self.base_model(x)
        return torch.stack([head(x) for head in self.policy_heads], dim=0)


class EncoderModel(nn.Module):
    def __init__(self, input_size, layers, latent_dim, use_orthogonal_init=True):
        super().__init__()
        self.encoder = make_fc([input_size] + list(layers) + [latent_dim], use_orthogonal_init=use_orthogonal_init)
        self.latent_dim = latent_dim
    
    def forward(self, x):
        return self.encoder(x)


class DecoderModel(nn.Module):
    def __init__(self, latent_dim, base_layers, head_layers, observation_output_sizes, action_output_sizes, use_orthogonal_init=True):
        super().__init__()
        self.decoder_base = make_fc([latent_dim] + list(base_layers), final_activation=nn.ReLU, use_orthogonal_init=use_orthogonal_init)
        self.base_dim = base_layers[-1]
        if observation_output_sizes is None:
            observation_output_sizes = []
        if action_output_sizes is None:
            action_output_sizes = []
        self.num_decoded_observations = len(observation_output_sizes)
        self.num_decoded_actions = len(action_output_sizes)
        output_sizes = observation_output_sizes + action_output_sizes
        self.decoder_heads = nn.ModuleList(
            [make_fc([self.base_dim] + list(head_layers) + [output_size], use_orthogonal_init=use_orthogonal_init) for output_size in output_sizes]
        )
    
    def forward(self, z):
        x = self.decoder_base(z)
        outputs = [head(x) for head in self.decoder_heads]
        observation_outputs = torch.stack(outputs[:self.num_decoded_observations], dim=0) if self.num_decoded_observations > 0 else None
        action_outputs = torch.stack(outputs[self.num_decoded_observations:], dim=0) if self.num_decoded_actions > 0 else None
        return observation_outputs, action_outputs


class EncoderDecoderModel(nn.Module):
    def __init__(
            self,
            input_size,
            encoder_layers,
            latent_dim,
            decoder_base_layers,
            decoder_head_layers,
            observation_output_sizes,
            action_output_sizes,
            use_orthogonal_init=True,
        ):
        super().__init__()
        self.encoder = EncoderModel(input_size, encoder_layers, latent_dim, use_orthogonal_init=use_orthogonal_init)
        self.decoder = DecoderModel(latent_dim, decoder_base_layers, decoder_head_layers, observation_output_sizes, action_output_sizes, use_orthogonal_init=use_orthogonal_init)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)
