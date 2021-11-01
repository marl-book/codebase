import numpy as np
import torch
from torch import nn
from typing import List


class MultiAgentFCNetwork(nn.Module):
    def _init_layer(self, m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m

    def _make_fc(self, dims, activation=nn.ReLU, final_activation=None):
        mods = []

        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(activation())
            mods.append(self._init_layer(nn.Linear(h_sizes[i], h_sizes[i + 1])))

        if final_activation:
            mods.append(final_activation())

        return nn.Sequential(*mods)

    def __init__(self, input_sizes, idims, output_sizes):
        super().__init__()

        n_agents = len(input_sizes)
        self.models = nn.ModuleList()

        for in_size, out_size in zip(input_sizes, output_sizes):
            dims = [in_size] + idims + [out_size]
            self.models.append(self._make_fc(dims))

    def forward(self, inputs: List[torch.Tensor]):
        futures = [
            torch.jit.fork(model, inputs[i]) for i, model in enumerate(self.models)
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        return results


class MultiAgentSEPSNetwork(nn.Module):
    def _init_layer(self, m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m

    def _make_fc(self, dims, activation=nn.ReLU, final_activation=None):
        mods = []

        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(activation())
            mods.append(self._init_layer(nn.Linear(h_sizes[i], h_sizes[i + 1])))

        if final_activation:
            mods.append(final_activation())

        return nn.Sequential(*mods)

    def __init__(self, input_sizes, idims):
        super().__init__()

        self.seps_size = len(input_sizes)
        self.independent = nn.ModuleList()

        for size in input_sizes:
            dims = [size] + idims
            self.independent.append(self._make_fc(dims))

    def forward(self, inputs, seps_indices):

        inputs = torch.stack(inputs)
        out = torch.stack([net(inputs) for net in self.independent])
        # No parallel envs, so needed to change from 3 --> 2
        if inputs[0].dim() == 2:
            seps_indices = seps_indices.T.unsqueeze(0).unsqueeze(-1).unsqueeze(2)
            seps_indices = seps_indices.expand(1, *out.shape[1:])
        else:
            seps_indices = (
                seps_indices.T.unsqueeze(0).unsqueeze(-1).expand(1, *out.shape[1:])
            )

        out = out.gather(0, seps_indices).split(1, dim=1)

        out = [x.squeeze(0).squeeze(0) for x in out]

        return out


class MultiAgentSEPSNetworkParallel(nn.Module):
    def _init_layer(self, m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m

    def _make_fc(self, dims, activation=nn.ReLU, final_activation=None):
        mods = []

        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(activation())
            mods.append(self._init_layer(nn.Linear(h_sizes[i], h_sizes[i + 1])))

        if final_activation:
            mods.append(final_activation())

        return nn.Sequential(*mods)

    def __init__(self, input_sizes, idims):
        super().__init__()

        self.seps_size = len(input_sizes)
        self.independent = nn.ModuleList()

        for size in input_sizes:
            dims = [size] + idims
            self.independent.append(self._make_fc(dims))

    def forward(self, inputs, seps_indices):

        inputs = torch.stack(inputs)
        out = torch.stack([net(inputs) for net in self.independent])
        if inputs[0].dim() == 3:
            seps_indices = seps_indices.T.unsqueeze(0).unsqueeze(-1).unsqueeze(2)
            seps_indices = seps_indices.expand(1, *out.shape[1:])
        else:
            seps_indices = (
                seps_indices.T.unsqueeze(0).unsqueeze(-1).expand(1, *out.shape[1:])
            )

        out = out.gather(0, seps_indices).split(1, dim=1)

        out = [x.squeeze(0).squeeze(0) for x in out]

        return out
