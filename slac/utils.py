import math

import torch
from torch import nn


def create_feature_actions(feature_, action_):
    N = feature_.size(0)
    # Flatten sequence of features.
    f = feature_[:, :-1].view(N, -1)
    n_f = feature_[:, 1:].view(N, -1)
    # Flatten sequence of actions.
    a = action_[:, :-1].view(N, -1)
    n_a = action_[:, 1:].view(N, -1)
    # Concatenate feature and action.
    fa = torch.cat([f, a], dim=-1)
    n_fa = torch.cat([n_f, n_a], dim=-1)
    return fa, n_fa


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


def build_mlp(
    input_dim,
    output_dim,
    hidden_units=[64, 64],
    hidden_activation=nn.Tanh(),
    output_activation=None,
):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_gaussian_log_prob(log_std, noise):
    return (-0.5 * noise.pow(2) - log_std).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_std.size(-1)


def calculate_log_pi(log_std, noise, action):
    gaussian_log_prob = calculate_gaussian_log_prob(log_std, noise)
    return gaussian_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(mean, log_std):
    noise = torch.randn_like(mean)
    action = torch.tanh(mean + noise * log_std.exp())
    return action, calculate_log_pi(log_std, noise, action)


def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
