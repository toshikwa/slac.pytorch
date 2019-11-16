import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from network.base import BaseNetwork


class Gaussian(BaseNetwork):
    ''' Mapping from features to (diagonal) Gaussian distributions. '''

    def __init__(self, input_dim, hidden_dim, output_dim, std=None):
        super(Gaussian, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2*output_dim if std is None else output_dim)
        )

        self.std = std

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x,  dim=-1)

        x = self.net(x)
        if self.scale:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        return Normal(loc=mean, scale=std)


class ConstantGaussian(BaseNetwork):
    ''' Gaussian distributions. '''

    def __init__(self, output_dim, std=1.0):
        super(ConstantGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        loc = torch.zeros((x.size(0), self.output_dim)).to(x)
        scale = torch.ones((x.size(0), self.output_dim)).to(x) * self.scale
        return Normal(loc=loc, scale=scale)


class Decoder(BaseNetwork):
    ''' Mapping from latent vectors to images. '''

    def __init__(self, feature_dim=256, latent1_dim=32, latent2_dim=256,
                 output_dim=3, std=1.0):
        super(Decoder, self).__init__()
        self.std = std

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(latent1_dim + latent2_dim, feature_dim, 4),
            nn.LeakyReLU(),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(feature_dim, feature_dim//2, 3, 2, 1, 1),
            nn.LeakyReLU(),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(feature_dim//2, feature_dim//4, 3, 2, 1, 1),
            nn.LeakyReLU(),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(feature_dim//4, feature_dim//8, 3, 2, 1, 1),
            nn.LeakyReLU(),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(feature_dim//8, output_dim, 5, 2, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x,  dim=-1)

        num_batches, num_sequences, latent_dim = x.size()
        x = x.view(num_batches * num_sequences, latent_dim, 1, 1)
        x = self.net(x)
        x.view(num_batches, num_sequences, *x.size())
        return Normal(loc=x, scale=torch.ones_like(x) * self.std)


class Encoder(BaseNetwork):
    ''' Mapping from images to features. '''

    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, feature_dim//8, 5, 2, 2),
            nn.LeakyReLU(),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(feature_dim//8, feature_dim//4, 3, 2, 1),
            nn.LeakyReLU(),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(feature_dim//4, feature_dim//2, 3, 2, 1),
            nn.LeakyReLU(),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(feature_dim//2, feature_dim, 3, 2, 1),
            nn.LeakyReLU(),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(feature_dim, feature_dim, 4),
            nn.LeakyReLU()
        )

    def forward(self, x):
        num_batches, num_sequences, C, H, W = x.size()
        x = x.view(num_batches * num_sequences, C, H, W)
        x = self.net(x)
        x = x.view(num_batches, num_sequences, -1)

        return x


class LatentNetwork(BaseNetwork):

    def __init__(self, observation_shape, action_shape, feature_dim=256,
                 latent1_dim=32, latent2_dim=256, kl_analytic=True,
                 model_reward=False, reward_std=None):
        super(LatentNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.feature_dim = feature_dim
        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim
        self.kl_analytic = kl_analytic
        self.model_reward = model_reward

        # Prior distributions.
        self.latent1_initial_prior = ConstantGaussian(latent1_dim)
        self.latent1_prior =\
            Gaussian(latent2_dim+action_shape[0], 256, latent1_dim)
        self.latent2_initial_prior =\
            Gaussian(latent1_dim, 256, latent2_dim)
        self.latent2_prior =\
            Gaussian(latent1_dim+latent2_dim+action_shape[0], 256, latent2_dim)

        # Mappings from feature to posterior distributions.
        self.latent1_initial_posterior =\
            Gaussian(feature_dim, 256, latent1_dim)
        self.latent1_posterior =\
            Gaussian(feature_dim+latent2_dim+action_shape[0], 256, latent1_dim)
        self.latent2_initial_posterior = self.latent2_initial_prior
        self.latent2_posterior = self.latent2_prior

        # Mappings from images to features.
        self.encoder = Encoder(observation_shape[0], feature_dim)
        # Mappings from latent vectors to images.
        self.decoder = Decoder(
            feature_dim, latent1_dim, latent2_dim, observation_shape[0],
            std=np.sqrt(0.1))

        if self.model_reward:
            self.reward_predictor = Gaussian(
                2*latent1_dim+2*latent2_dim+action_shape[0], 256, 1,
                std=reward_std)
        else:
            self.reward_predictor = None

    @property
    def state_size(self):
        return self.latent1_dim + self.latent2_dim

    def compute_loss(self, images, actions, start_flag, rewards=None,
                     latent_posterior_samples=None):
        outputs = {}
        sequence_length = images.size(1)

        if latent_posterior_samples is None:
            posterior_samples, posterior_dists =\
                self.sample_posterior(images, actions)
            latent1_posterior_samples, latent2_posterior_samples =\
                posterior_samples
            latent1_posterior_dists, latent2_posterior_dists =\
                posterior_dists

        # For visualization.
        (latent1_prior_samples, latent2_prior_samples), _ =\
            self.sample_prior_or_posterior(actions)
        (latent1_cond_prior_samples, latent2_cond_prior_samples), _ =\
            self.sample_prior_or_posterior(actions, images=images[:, :1])

        prior_samples, prior_dists = self.sample_prior(images, actions)
        _, latent2_prior_samples = prior_samples
        latent1_prior_dists, latent2_prior_dists = prior_dists

        latent1_kld = 0.0
        if self.kl_analytic:
            for post, pri in zip(latent1_posterior_dists, latent1_prior_dists):
                latent1_kld += kl_divergence(post, pri)
        else:
            latent1_kld = 0.0
            for i, (post, pri) in enumerate(zip(
                    latent1_posterior_dists, latent1_prior_dists)):
                latent1_kld += post.log_prob(latent1_posterior_samples[:, i])\
                    - pri.log_prob(latent1_posterior_samples[:, i])

        latent2_kld = 0.0
        if self.latent2_posterior != self.latent2_prior:
            if self.kl_analytic:
                for post, pri in zip(
                        latent2_posterior_dists, latent2_prior_dists):
                    latent2_kld += kl_divergence(post, pri)
            else:
                for i, (post, pri) in enumerate(zip(
                        latent2_posterior_dists, latent2_prior_dists)):
                    latent2_kld += torch.mean(
                        post.log_prob(latent2_posterior_samples[:, i])
                        - pri.log_prob(latent2_posterior_samples[:, i]))

        kld = latent1_kld + latent2_kld

        reconstruct_dists = self.decoder(
            [latent1_posterior_samples, latent2_posterior_samples])
        reconstruction_log_probs = reconstruct_dists.log_prob(images)
        reconstruction_log_probs = torch.sum(reconstruction_log_probs, dim=1)
        reconstruction_log_probs = torch.mean(reconstruction_log_probs)
        reconstruction_error = torch.sum(
            (images - reconstruct_dists.loc).pow(2),
            dim=list(range(-len(images.shape), 0)))
        reconstruction_error = torch.maen(reconstruction_error)

        elbo = reconstruction_log_probs - latent1_kld - latent2_kld

        outputs.update({
            'latent1_kld': latent1_kld,
            'latent2_kld': latent2_kld,
            'kld': kld,
            'log_prob': reconstruction_log_probs,
            'reconstruction_error': reconstruction_error
        })

        if self.model_reward:
            reward_dists = self.reward_predictor([
                latent1_posterior_samples[:, :sequence_length],
                latent2_posterior_samples[:, :sequence_length],
                actions[:, :sequence_length],
                latent1_posterior_samples[:, 1:sequence_length + 1],
                latent2_posterior_samples[:, 1:sequence_length + 1]])
            reward_log_probs =\
                reward_dists.log_prob(rewards[:, :sequence_length])
            reward_log_probs[:, sequence_length] = 0
            reward_log_probs = torch.sum(reward_log_probs, axis=1)
            reward_reconstruction_error =\
                (rewards[:, :sequence_length] - reward_dists.loc).pow(2)

            reward_reconstruction_error[:, sequence_length] = 0
            reward_reconstruction_error =\
                torch.sum(reward_reconstruction_error, axis=1)
            elbo += torch.mean(reward_log_probs)

        loss = -elbo

        posterior_images = reconstruct_dists.mean()
        prior_images = self.decoder([
            latent1_prior_samples, latent2_prior_samples]).mean()
        conditional_prior_images = self.decoder([
            latent1_cond_prior_samples,
            latent2_cond_prior_samples]).mean()

        outputs.update({
            'elbo': elbo,
            'images': images,
            'posterior_images': posterior_images,
            'prior_images': prior_images,
            'conditional_prior_images': conditional_prior_images,
        })

        return loss, outputs

    def sample_prior_or_posterior(self, actions, images=None):
        sequence_length = images.size(1)
        actions = actions[:, :sequence_length-1]

        if images is not None:
            features = self.encoder(images)

            # Swap batch and time axes.
            features = torch.transpose(features, 0, 1)
        actions = torch.transpose(actions, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(sequence_length):
            is_conditional = images is not None
            if t == 0:
                if is_conditional:
                    latent1_dist = self.latent1_initial_posterior(features[t])
                else:
                    latent1_dist = self.latent1_initial_prior(features[t])
                latent1_sample = latent1_dist.sample()

                if is_conditional:
                    latent2_dist =\
                        self.latent2_initial_posterior(latent1_sample)
                else:
                    latent2_dist = self.latent2_initial_prior(latent1_sample)
                latent2_sample = latent2_dist.sample()

            else:
                if is_conditional:
                    latent1_dist = self.latent1_posterior(
                        [features[t], latent2_samples[t-1], actions[t-1]])
                else:
                    latent1_dist = self.latent1_prior(
                        [latent2_samples[t-1], actions[t-1]])
                latent1_sample = latent1_dist.sample()

                if is_conditional:
                    latent2_dist = self.latent2_posterior(
                        [latent1_sample, latent2_samples[t-1], actions[t-1]])
                else:
                    latent2_dist = self.latent2_prior(
                        [latent1_sample, latent2_samples[t-1], actions[t-1]])
                latent2_sample = latent2_dist.sample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)

    def sample_prior(self, images, actions):
        sequence_length = images.size(1)
        actions = actions[:, :sequence_length-1]

        # Swap batch and time axes.
        actions = torch.transpose(actions, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(sequence_length):
            if t == 0:
                latent1_dist = self.latent1_initial_prior(actions[t])
                latent1_sample = latent1_dist.sample()
                latent2_dist = self.latent2_initial_prior(latent1_sample)
                latent2_sample = latent2_dist.sample()

            else:
                latent1_dist = self.latent1_prior(
                    [latent2_samples[t-1], actions[t-1]])
                latent1_sample = latent1_dist.sample()
                latent2_dist = self.latent2_prior(
                    [latent1_sample, latent2_samples[t-1], actions[t-1]])
                latent2_sample = latent2_dist.sample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent1_dists)

    def sample_posterior(self, images, actions, features=None):
        sequence_length = images.size(1)
        actions = actions[:, :sequence_length-1]

        if features is None:
            features = self.encoder(images)

        # Swap batch and time axes.
        features = torch.transpose(features, 0, 1)
        actions = torch.transpose(actions, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(sequence_length):
            if t == 0:
                latent1_dist = self.latent1_initial_posterior(features[t])
                latent1_sample = latent1_dist.sample()
                latent2_dist = self.latent2_initial_posterior(latent1_sample)
                latent2_sample = latent2_dist.sample()

            else:
                latent1_dist = self.latent1_posterior(
                    [features[t], latent2_samples[t-1], actions[t-1]])
                latent1_sample = latent1_dist.sample()
                latent2_dist = self.latent2_posterior(
                    [latent1_sample, latent2_samples[t-1], actions[t-1]])
                latent2_sample = latent2_dist.sample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent1_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)
