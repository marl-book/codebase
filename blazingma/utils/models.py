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
