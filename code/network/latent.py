import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from network.base import BaseNetwork, weights_init_xavier


class Gaussian(BaseNetwork):

    def __init__(self, input_dim, hidden_dim, output_dim, std=None):
        super(Gaussian, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2*output_dim if std is None else output_dim)
        ).apply(weights_init_xavier)
        self.std = std

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x,  dim=-1)

        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        return Normal(loc=mean, scale=std)


class ConstantGaussian(BaseNetwork):
    ''' Constant gaussian distributions. '''

    def __init__(self, output_dim, std=1.0):
        super(ConstantGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        # Output (num_batches, output_dim) size of Gaussians.
        mean = torch.zeros((x.size(0), self.output_dim)).to(x)
        std = torch.ones((x.size(0), self.output_dim)).to(x) * self.std
        return Normal(loc=mean, scale=std)


class Decoder(BaseNetwork):
    ''' Mapping from latent vectors to images. '''

    def __init__(self, input_dim=288, output_dim=3, std=1.0):
        super(Decoder, self).__init__()
        self.std = std

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, 3, 5, 2, 2, 1),
            nn.LeakyReLU()
        ).apply(weights_init_xavier)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x,  dim=-1)

        num_batches, num_sequences, latent_dim = x.size()
        x = x.view(num_batches * num_sequences, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(num_batches, num_sequences, C, W, H)
        return Normal(loc=x, scale=torch.ones_like(x) * self.std)


class Encoder(BaseNetwork):
    ''' Deterministic mapping from images to features. '''

    def __init__(self, input_dim=3, output_dim=256):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU()
        ).apply(weights_init_xavier)

    def forward(self, x):
        num_batches, num_sequences, C, H, W = x.size()
        x = x.view(num_batches * num_sequences, C, H, W)
        x = self.net(x)
        x = x.view(num_batches, num_sequences, -1)

        return x


class LatentNetwork(BaseNetwork):

    def __init__(self, observation_shape, action_shape, feature_dim=256,
                 latent1_dim=32, latent2_dim=256, hidden_dim=256,
                 kl_analytic=True):
        super(LatentNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.feature_dim = feature_dim
        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim
        self.hidden_dim = hidden_dim
        self.kl_analytic = kl_analytic

        # p(z1(0)) = N(0, I)
        self.latent1_init_prior = ConstantGaussian(latent1_dim)
        # p(z2(0) | z1(0))
        self.latent2_init_prior = Gaussian(
            latent1_dim, hidden_dim, latent2_dim)

        # p(z1(t+1) | z2(t), a(t))
        self.latent1_prior = Gaussian(
            latent2_dim + action_shape[0], hidden_dim, latent1_dim)
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_prior = Gaussian(
            latent1_dim + latent2_dim + action_shape[0], hidden_dim,
            latent2_dim)

        # NOTE: We encode x as the feature vector to share convolutional
        # part of the network with the policy.

        # q(z1(0) | feat(0))
        self.latent1_init_posterior = Gaussian(
            feature_dim, hidden_dim, latent1_dim)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.latent2_init_posterior = self.latent2_init_prior

        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.latent1_posterior = Gaussian(
            feature_dim + latent2_dim + action_shape[0], hidden_dim,
            latent1_dim)
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_posterior = self.latent2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward_predictor = Gaussian(
            2 * latent1_dim + 2 * latent2_dim + action_shape[0], hidden_dim, 1)

        # feat(t) = x(t) : This encoding is performed deterministicly.
        self.encoder = Encoder(observation_shape[0], feature_dim)
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            latent1_dim + latent2_dim, observation_shape[0],
            std=np.sqrt(0.1))

    @property
    def state_size(self):
        return self.latent1_dim + self.latent2_dim

    def calc_loss(self, images, features, actions, rewards, dones):
        '''
        Args:
            images   : (N, S, 3, 64, 64) shaped array.
            features : (N, S, 256) shaped array.
            actions  : (N, S-1, |A|) shaped array.
            rewards  : (N, S-1, 1) shaped array.
            dones    : (N, S-1, 1) shaped array.
        '''

        num_sequences = actions.size(1) + 1

        # Sample from posterior dynamics.
        # (N, S) x 2 samples, (S, N) x 2 dists
        (latent1_post_samples, latent2_post_samples),\
            (latent1_post_dists, latent2_post_dists) =\
            self.sample_posterior(features, actions)

        # Sample from prior dynamics.
        # (N, S) x 2 samples, (S, N) x 2 dists
        (latent1_pri_samples, latent2_pri_samples),\
            (latent1_pri_dists, latent2_pri_dists) = self.sample_prior(actions)

        # Calculate KL divergence between prior and posterior of z1.
        kld = 0.0
        if self.kl_analytic:
            for i in range(num_sequences):
                kld += torch.mean(kl_divergence(
                    latent1_post_dists[i], latent1_pri_dists[i]))
        else:
            for i in range(num_sequences):
                kld += torch.mean(
                    latent1_post_dists[i].log_prob(latent1_post_samples[:, i])
                    - latent1_pri_dists[i].log_prob(latent1_post_samples[:, i])
                    )

        # NOTE: We use the same distribution for prior and posterior of z2,
        # whose KL diverhence are 0.

        # p(x(t) | z1(t), z2(t))
        image_dists = self.decoder(
            [latent1_post_samples, latent2_post_samples])
        # log likelihood of x(t).
        log_likelihoods = image_dists.log_prob(images).sum(dim=1).mean()

        # Reconstruction errors for debugging.
        reconst_errors = (images-image_dists.loc).pow(2).mean(
            dim=(0, 1)).sum().detach().item()

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        reward_dists = self.reward_predictor([
            latent1_post_samples[:, :num_sequences-1],
            latent2_post_samples[:, :num_sequences-1],
            actions, latent1_post_samples[:, 1:num_sequences],
            latent2_post_samples[:, 1:num_sequences]])
        # Log likelihood of r(t) with masking where done = True.
        reward_log_likelihood = reward_dists.log_prob(rewards) * (1.0-dones)
        reward_log_likelihood = reward_log_likelihood.sum(dim=1).mean()

        # Reconstruction errors for debugging.
        reward_reconst_errors = (rewards-reward_dists.loc).pow(2) * (1.0-dones)
        reward_reconst_errors = reward_reconst_errors.mean(
            dim=(0, 1)).sum().detach().item()

        loss = kld - log_likelihoods - reward_log_likelihood

        return loss, reconst_errors, reward_reconst_errors, image_dists.loc

    def sample_prior(self, actions):
        ''' Sample from prior dynamics.
        Args:
            actions : (N, S-1, |A|) shaped array.
        '''
        num_sequences = actions.size(1) + 1

        # (S-1, N, |A|)
        actions = torch.transpose(actions, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences):
            if t == 0:
                # p(z1(0)) = N(0, I)
                latent1_dist = self.latent1_init_prior(actions[t])
                latent1_sample = latent1_dist.sample()
                # p(z2(0) | z1(0))
                latent2_dist = self.latent2_init_prior(latent1_sample)
                latent2_sample = latent2_dist.sample()

            else:
                # p(z1(t) | z2(t-1), a(t-1))
                latent1_dist = self.latent1_prior(
                    [latent2_samples[t-1], actions[t-1]])
                latent1_sample = latent1_dist.sample()
                # p(z2(t) | z1(t), z2(t-1), a(t-1))
                latent2_dist = self.latent2_prior(
                    [latent1_sample, latent2_samples[t-1], actions[t-1]])
                latent2_sample = latent2_dist.sample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        # (N, S, 32)
        latent1_samples = torch.stack(latent1_samples, dim=1)
        # (N, S, 256)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)

    def sample_posterior(self, features, actions):
        ''' Sample from posterior dynamics.
        Args:
            features: (N, S, 256) shaped array.
            actions : (N, S-1, |A|) shaped array.
        '''
        num_sequences = actions.size(1) + 1

        # (S, N, 256)
        features = torch.transpose(features, 0, 1)
        # (S-1, N, |A|)
        actions = torch.transpose(actions, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences):
            if t == 0:
                # q(z1(0) | feat(0))
                latent1_dist = self.latent1_init_posterior(features[t])
                latent1_sample = latent1_dist.sample()
                # q(z2(0) | z1(0))
                latent2_dist = self.latent2_init_posterior(latent1_sample)
                latent2_sample = latent2_dist.sample()
            else:
                # q(z1(t) | feat(t), z2(t-1), a(t-1))
                latent1_dist = self.latent1_posterior(
                    [features[t], latent2_samples[t-1], actions[t-1]])
                latent1_sample = latent1_dist.sample()
                # q(z2(t) | z1(t), z2(t-1), a(t-1))
                latent2_dist = self.latent2_posterior(
                    [latent1_sample, latent2_samples[t-1], actions[t-1]])
                latent2_sample = latent2_dist.sample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        # (N, S, 32)
        latent1_samples = torch.stack(latent1_samples, dim=1)
        # (N, S, 256)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)
