from collections import deque
import numpy as np
import torch
from torch.distributions.kl import kl_divergence


def create_feature_actions(features_seq, actions_seq):
    N = features_seq.size(0)

    # sequence of features
    f = features_seq[:, :-1].view(N, -1)
    n_f = features_seq[:, 1:].view(N, -1)
    # sequence of actions
    a = actions_seq[:, :-1].view(N, -1)
    n_a = actions_seq[:, 1:].view(N, -1)

    # feature_actions
    fa = torch.cat([f, a], dim=-1)
    n_fa = torch.cat([n_f, n_a], dim=-1)

    return fa, n_fa


def calc_kl_divergence(p_list, q_list):
    assert len(p_list) == len(q_list)

    kld = 0.0
    for i in range(len(p_list)):
        # (N, L) shaped array of kl divergences.
        _kld = kl_divergence(p_list[i], q_list[i])
        # Average along batches, sum along sequences and elements.
        kld += _kld.mean(dim=0).sum()

    return kld


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
