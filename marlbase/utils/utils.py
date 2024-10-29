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


def compute_nstep_returns(rewards, done, next_values, nsteps, gamma):
    """
    Computed n-step returns
    :param rewards: tensor of shape (ep_length, batch_size, n_agents)
    :param done: tensor of shape (ep_length, batch_size, n_agents)
    :param next_values: tensor of shape (ep_length, batch_size, n_agents)
    :param nsteps: number of steps to bootstrap
    :param gamma: discount factor
    :return: tensor of shape with returns (ep_length, batch_size, n_agents)
    """
    ep_length = rewards.size(0)
    nstep_values = torch.zeros_like(rewards)
    for t_start in range(ep_length):
        nstep_return_t = torch.zeros_like(rewards[0])
        for step in range(nsteps + 1):
            t = t_start + step
            if t >= ep_length:
                # episode has ended
                break
            elif step == nsteps:
                # last n-step value --> bootstrap from the next value
                nstep_return_t += gamma**step * next_values[t] * (1 - done[t])
            else:
                nstep_return_t += gamma**step * rewards[t] * (1 - done[t])
        nstep_values[t_start] = nstep_return_t
    return nstep_values
