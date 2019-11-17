import torch
from torch.distributions import Normal
from rltorch.network import create_linear_network

from network.base import BaseNetwork


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, input_dim, output_dim, hidden_units=[256, 256],
                 initializer='xavier'):
        super(GaussianPolicy, self).__init__()

        # NOTE: Conv layers are shared with Encoder.
        self.net = create_linear_network(
            input_dim, output_dim*2, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x,  dim=-1)

        mean, log_std = torch.chunk(self.net(x), 2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, x):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(x)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)
