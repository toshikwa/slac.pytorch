import math

import torch
from torch import nn


def create_feature_actions(features_, actions_):
    N = features_.size(0)

    # sequence of features
    f = features_[:, :-1].view(N, -1)
    n_f = features_[:, 1:].view(N, -1)
    # sequence of actions
    a = actions_[:, :-1].view(N, -1)
    n_a = actions_[:, 1:].view(N, -1)

    # feature_actions
    fa = torch.cat([f, a], dim=-1)
    n_fa = torch.cat([n_f, n_a], dim=-1)

    return fa, n_fa


def kl_divergence_loss(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow(2)
    t1 = ((p_mean - q_mean) / q_std).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - torch.log(var_ratio)).mean(dim=0).sum()


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


def calculate_gaussian_log_prob(log_stds, noises):
    return (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = calculate_gaussian_log_prob(log_stds, noises)
    return gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def rsample(means, stds):
    return means + torch.rand_like(stds) * stds


def reparameterize(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)
