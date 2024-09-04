import torch


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


def to_onehot(tensor, n_dims):
    """
    Convert tensor of indices to one-hot representation
    :param tensor: tensor of indices (batch_size, ..., 1)
    :param n_dims: number of dimensions
    :return: one-hot representation (batch_size, ..., n_dims)
    """
    onehot = torch.zeros(tensor.shape + (n_dims,), device=tensor.device)
    return onehot.scatter(-1, tensor.unsqueeze(-1), 1)
