from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn


def orthogonal_init(m):
    nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
    nn.init.constant_(m.bias.data, 0)
    return m


class FCNetwork(nn.Module):
    def __init__(
        self, dims, activation=nn.ReLU, final_activation=None, use_orthogonal_init=True
    ):
        """
        Create fully-connected network
        :param dims: list of dimensions for the network
        :param activation: activation function to use
        :param final_activation: activation function to use on output (if any)
        :param use_orthogonal_init: whether to use orthogonal initialization
        :return: sequential network
        """
        super().__init__()
        mods = []

        input_size = dims[0]
        h_sizes = dims[1:]

        init_fn = orthogonal_init if use_orthogonal_init else lambda x: x

        mods = [init_fn(nn.Linear(input_size, h_sizes[0]))]
        for i in range(len(h_sizes) - 1):
            mods.append(activation())
            mods.append(init_fn(nn.Linear(h_sizes[i], h_sizes[i + 1])))

        if final_activation:
            mods.append(final_activation())

        self.network = nn.Sequential(*mods)

    def init_hiddens(self, batch_size, device):
        return None

    def forward(self, x, h=None):
        return self.network(x), None


class RNNNetwork(nn.Module):
    def __init__(
        self,
        dims,
        rnn=nn.GRU,
        activation=nn.ReLU,
        final_activation=None,
        use_orthogonal_init=True,
    ):
        """
        Create recurrent network (last layer is fully connected)
        :param dims: list of dimensions for the network
        :param activation: activation function to use
        :param final_activation: activation function to use on output (if any)
        :param use_orthogonal_init: whether to use orthogonal initialization
        :return: sequential network
        """
        super().__init__()
        assert (
            len(dims) > 2
        ), "Need at least 3 dimensions for RNN (1 input dim, >= 1 hidden dim, 1 output dim)"

        assert rnn in [nn.GRU, nn.LSTM], "Only GRU and LSTM are supported"

        input_size = dims[0]
        rnn_hiddens = dims[1:-1]
        rnn_hidden_size = rnn_hiddens[0]
        assert all(
            rnn_hidden_size == h for h in rnn_hiddens
        ), "Expect same hidden size across all RNN layers"
        output_size = dims[-1]

        self.rnn = rnn(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=len(rnn_hiddens),
            batch_first=False,
        )
        self.activation = activation()
        self.final_layer = nn.Linear(rnn_hidden_size, output_size)
        if use_orthogonal_init:
            self.final_layer = orthogonal_init(self.final_layer)

        self.final_activation = final_activation

    def init_hiddens(self, batch_size, device):
        return torch.zeros(
            self.rnn.num_layers,
            batch_size,
            self.rnn.hidden_size,
            device=device,
        )

    def forward(self, x, h=None):
        assert x.dim() == 3, "Expect input to be 3D tensor (seq_len, batch, input_size)"
        assert (
            h is None or h.dim() == 3
        ), "Expect hidden state to be 3D tensor (num_layers, batch, hidden_size)"
        x, h = self.rnn(x, h)
        x = self.activation(x)
        x = self.final_layer(x)
        if self.final_activation:
            x = self.final_activation(x)
        return x, h


def make_network(
    dims,
    use_rnn=False,
    rnn=nn.GRU,
    activation=nn.ReLU,
    final_activation=None,
    use_orthogonal_init=True,
):
    if use_rnn:
        return RNNNetwork(dims, rnn, activation, final_activation, use_orthogonal_init)
    else:
        return FCNetwork(dims, activation, final_activation, use_orthogonal_init)


class MultiAgentIndependentNetwork(nn.Module):
    def __init__(
        self,
        input_sizes,
        hidden_dims,
        output_sizes,
        use_rnn=False,
        use_orthogonal_init=True,
    ):
        super().__init__()
        assert len(input_sizes) == len(
            output_sizes
        ), "Expect same number of input and output sizes"
        self.independent = nn.ModuleList()

        for in_size, out_size in zip(input_sizes, output_sizes):
            dims = [in_size] + hidden_dims + [out_size]
            self.independent.append(
                make_network(
                    dims, use_rnn=use_rnn, use_orthogonal_init=use_orthogonal_init
                )
            )

    def forward(
        self,
        inputs: Union[List[torch.Tensor], torch.Tensor],
        hiddens: Optional[List[torch.Tensor]] = None,
    ):
        if hiddens is None:
            hiddens = [None] * len(inputs)
        futures = [
            torch.jit.fork(model, x, h)
            for model, x, h in zip(self.independent, inputs, hiddens)
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        outs = [x for x, _ in results]
        hiddens = [h for _, h in results]
        return outs, hiddens

    def init_hiddens(self, batch_size, device):
        return [model.init_hiddens(batch_size, device) for model in self.independent]


class MultiAgentSharedNetwork(nn.Module):
    def __init__(
        self,
        input_sizes,
        hidden_dims,
        output_sizes,
        sharing_indices,
        use_rnn=False,
        use_orthogonal_init=True,
    ):
        super().__init__()
        assert len(input_sizes) == len(
            output_sizes
        ), "Expect same number of input and output sizes"
        self.num_agents = len(input_sizes)

        if sharing_indices is True:
            self.sharing_indices = len(input_sizes) * [0]
        elif sharing_indices is False:
            self.sharing_indices = list(range(len(input_sizes)))
        else:
            self.sharing_indices = sharing_indices
        assert len(self.sharing_indices) == len(
            input_sizes
        ), "Expect same number of sharing indices as agents"

        self.num_networks = 0
        self.networks = nn.ModuleList()
        self.agents_by_network = []
        self.input_sizes = []
        self.output_sizes = []
        created_networks = set()
        for i in self.sharing_indices:
            if i in created_networks:
                # network already created
                continue

            # agent indices that share this network
            network_agents = [
                j for j, idx in enumerate(self.sharing_indices) if idx == i
            ]
            in_sizes = [input_sizes[j] for j in network_agents]
            in_size = in_sizes[0]
            assert all(
                idim == in_size for idim in in_sizes
            ), f"Expect same input sizes across all agents sharing network {i}"
            out_sizes = [output_sizes[j] for j in network_agents]
            out_size = out_sizes[0]
            assert all(
                odim == out_size for odim in out_sizes
            ), f"Expect same output sizes across all agents sharing network {i}"

            dims = [in_size] + hidden_dims + [out_size]
            self.networks.append(
                make_network(
                    dims, use_rnn=use_rnn, use_orthogonal_init=use_orthogonal_init
                )
            )
            self.agents_by_network.append(network_agents)
            self.input_sizes.append(in_size)
            self.output_sizes.append(out_size)
            self.num_networks += 1
            created_networks.add(i)

    def forward(
        self,
        inputs: Union[List[torch.Tensor], torch.Tensor],
        hiddens: Optional[List[torch.Tensor]] = None,
    ):
        assert all(
            x.dim() == 3 for x in inputs
        ), "Expect each agent input to be 3D tensor (seq_len, batch, input_size)"
        assert hiddens is None or all(
            x is None or x.dim() == 3 for x in hiddens
        ), "Expect hidden state to be 3D tensor (num_layers, batch, hidden_size)"

        batch_size = inputs[0].size(1)
        assert all(
            x.size(1) == batch_size for x in inputs
        ), "Expect all agent inputs to have same batch size"

        # group inputs and hiddens by network
        network_inputs = []
        network_hiddens = []
        for agent_indices in self.agents_by_network:
            net_inputs = [inputs[i] for i in agent_indices]
            if hiddens is None or all(h is None for h in hiddens):
                net_hiddens = None
            else:
                net_hiddens = [hiddens[i] for i in agent_indices]
            network_inputs.append(torch.cat(net_inputs, dim=1))
            network_hiddens.append(
                torch.cat(net_hiddens, dim=1) if net_hiddens is not None else None
            )

        # forward through networks
        futures = [
            torch.jit.fork(network, x, h)
            for network, x, h in zip(self.networks, network_inputs, network_hiddens)
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        outs = [x.split(batch_size, dim=1) for x, _ in results]
        hiddens = [
            h.split(batch_size, dim=1) if h is not None else None for _, h in results
        ]

        # group outputs by agents
        agent_outputs = []
        agent_hiddens = []
        self.idx_by_network = [0] * self.num_networks
        for network_idx in self.sharing_indices:
            idx_within_network = self.idx_by_network[network_idx]
            agent_outputs.append(outs[network_idx][idx_within_network])
            if hiddens[network_idx] is not None:
                agent_hiddens.append(hiddens[network_idx][idx_within_network])
            else:
                agent_hiddens.append(None)
            self.idx_by_network[network_idx] += 1
        return agent_outputs, agent_hiddens

    def init_hiddens(self, batch_size, device):
        return [
            self.networks[network_idx].init_hiddens(batch_size, device)
            for network_idx in self.sharing_indices
        ]
