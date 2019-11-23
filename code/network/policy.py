import torch
from torch import nn
from torch.distributions import Normal

from network.base import BaseNetwork, create_linear_network,\
    weights_init_xavier


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, input_dim, output_dim, hidden_units=[256, 256],
                 initializer=weights_init_xavier):
        super(GaussianPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        self.net = create_linear_network(
            input_dim, output_dim*2, hidden_units=hidden_units,
            hidden_activation=nn.ReLU(), initializer=initializer)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x,  dim=-1)

        means, log_stds = torch.chunk(self.net(x), 2, dim=-1)
        log_stds = torch.clamp(
            log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return means, log_stds

    def sample(self, x):
        # Calculate Gaussian distribusion of (means, stds).
        means, log_stds = self.forward(x)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # Sample actions.
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # Calculate expectations of entropies.
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)
