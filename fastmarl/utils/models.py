from abc import ABC, abstractmethod
from typing import List

from einops import rearrange, repeat
import numpy as np
import torch
from torch import nn


def orthogonal_init(m):
    nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
    nn.init.constant_(m.bias.data, 0)
    return m


def make_fc(dims, activation=nn.ReLU, final_activation=None, use_orthogonal_init=True):
    mods = []

    input_size = dims[0]
    h_sizes = dims[1:]

    mods = [nn.Linear(input_size, h_sizes[0])]
    for i in range(len(h_sizes) - 1):
        mods.append(activation())
        layer = nn.Linear(h_sizes[i], h_sizes[i + 1])
        if use_orthogonal_init:
            mods.append(orthogonal_init(layer))
        else:
            mods.append(layer)

    if final_activation:
        mods.append(final_activation())

    return nn.Sequential(*mods)


class MultiAgentNetwork(ABC, nn.Module):
    def _make_fc(self, dims, activation=nn.ReLU, final_activation=None, use_orthogonal_init=True):
        return make_fc(dims, activation, final_activation, use_orthogonal_init)


class MultiAgentFCNetwork(MultiAgentNetwork):
    def __init__(self, input_sizes, idims, output_sizes, use_orthogonal_init=True):
        super().__init__()
        assert len(input_sizes) == len(output_sizes), "Expect same number of input and output sizes"
        self.independent = nn.ModuleList()

        for in_size, out_size in zip(input_sizes, output_sizes):
            dims = [in_size] + idims + [out_size]
            self.independent.append(self._make_fc(dims, use_orthogonal_init=use_orthogonal_init))

    def forward(self, inputs: List[torch.Tensor]):
        futures = [
            torch.jit.fork(model, x) for model, x in zip(self.independent, inputs)
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        return results


class MultiAgentSEPSNetwork(MultiAgentNetwork):
    def __init__(self, input_sizes, idims, output_sizes, seps_indices, use_orthogonal_init=True):
        super().__init__()
        assert len(input_sizes) == len(output_sizes), "Expect same number of input and output sizes"
        assert all(in_size == input_sizes[0] for in_size in input_sizes), "Expect same input sizes across all agents for shared network"
        assert all(out_size == output_sizes[0] for out_size in output_sizes), "Expect same output sizes across all agents for shared network"
        self.independent = nn.ModuleList()
        
        if seps_indices is True:
            self.seps_indices = len(input_sizes) * [0]
        elif seps_indices is False:
            self.seps_indices = list(range(len(input_sizes)))
        else:
            self.seps_indices = seps_indices
        self.seps_size = len(set(self.seps_indices))
        self.seps_indices = torch.tensor(self.seps_indices, dtype=torch.int64)

        self.out_size = output_sizes[0]

        for _ in range(self.seps_size):
            dims = [input_sizes[0]] + idims + [output_sizes[0]]
            self.independent.append(self._make_fc(dims, use_orthogonal_init=use_orthogonal_init))

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        batch_shape = inputs.shape[1:-1]
        # input shape must be: n_agents, n_step(?), parallel_envs, net_out_size (e.g. # actions)
        if inputs.dim() == 2: # no n-step and no parallel envs
            inputs = rearrange(inputs, "A O -> A 1 1 O")
        elif inputs.dim() == 3: # there no n-step here
            inputs = rearrange(inputs, "A E O -> A 1 E O")

        out = torch.stack([net(inputs) for net in self.independent])

        seps_indices = repeat(self.seps_indices.T, "A -> 1 A N E O", N=out.shape[-3], E=out.shape[-2], O=out.shape[-1])
        out = out.gather(0, seps_indices).split(1, dim=1)
        return [x.reshape(*batch_shape, -1) for x in out]
