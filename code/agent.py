import os
from time import time
from collections import deque
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from memory import LazyMemory
from network import LatentNetwork, GaussianPolicy, TwinnedQNetwork
from utils import calc_kl_divergence, grad_false, hard_update, soft_update,\
    update_params, RunningMeanStats


class SlacAgent:
    def __init__(self, env, log_dir, env_type='dm_control', num_steps=3000000,
                 batch_size=256, latent_batch_size=32, num_sequences=8,
                 action_repeat=4, lr=0.0003, latent_lr=0.0001, feature_dim=256,
                 latent1_dim=32, latent2_dim=256, hidden_units=[256, 256],
                 memory_size=1e5, gamma=0.99, tau=0.005, entropy_tuning=True,
                 ent_coef=0.2, grad_clip=None, updates_per_step=1,
                 start_steps=10000, training_log_interval=10,
                 learning_log_interval=100, target_update_interval=1,
                 eval_interval=50000, cuda=True, seed=0):

        self.env = env
        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape

        torch.manual_seed(seed)
        np.random.seed(seed)
        # self.env.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.latent = LatentNetwork(
            self.observation_shape, self.action_shape, feature_dim,
            latent1_dim, latent2_dim).to(self.device)

        self.policy = GaussianPolicy(
            num_sequences * feature_dim
            + (num_sequences-1) * self.action_shape[0],
            self.action_shape[0], hidden_units).to(self.device)

        self.critic = TwinnedQNetwork(
            latent1_dim+latent2_dim, self.action_shape[0], hidden_units
            ).to(self.device)
        self.critic_target = TwinnedQNetwork(
            latent1_dim+latent2_dim, self.action_shape[0], hidden_units
            ).to(self.device).eval()

        # Copy parameters of the learning network to the target network.
        hard_update(self.critic_target, self.critic)
        # Disable gradient calculations of the target network.
        grad_false(self.critic_target)

        # Policy is updated without the encoder.
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)
        self.latent_optim = Adam(self.latent.parameters(), lr=latent_lr)

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_shape)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = torch.tensor(ent_coef).to(self.device)

        self.memory = LazyMemory(
            memory_size, num_sequences, self.observation_shape,
            self.action_shape, self.device)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(training_log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_sequences = num_sequences
        self.action_repeat = action_repeat
        self.num_steps = num_steps
        self.tau = tau
        self.batch_size = batch_size
        self.latent_batch_size = latent_batch_size
        self.start_steps = start_steps
        self.gamma = gamma
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.training_log_interval = training_log_interval
        self.learning_log_interval = learning_log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps * self.action_repeat

    def is_policy(self, state_deque):
        return self.start_steps > self.steps * self.action_repeat and\
            len(state_deque) == state_deque.maxlen

    def deque_to_batch(self, state_deque, action_deque):
        # feature: (1, 256*S)
        state = (np.stack(state_deque, axis=0)/255.0).astype(np.float32)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.latent.encoder(state).view(1, -1)
        # action: (1, |A|*(S-1))
        action = np.stack(action_deque, axis=0)
        action = torch.FloatTensor(action).view(1, -1).to(self.device)
        # trajectory: (1, 256*S + |A|*(S-1))
        trajectory = torch.cat([feature, action], dim=-1)

        return trajectory

    def explore(self, state_deque, action_deque):
        # act with randomness
        trajectory = self.deque_to_batch(state_deque, action_deque)
        with torch.no_grad():
            action, _, _ = self.policy.sample(trajectory)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state_deque, action_deque):
        # act without randomness
        trajectory = self.deque_to_batch(state_deque, action_deque)
        with torch.no_grad():
            _, _, action = self.policy.sample(trajectory)
        return action.cpu().numpy().reshape(-1)

    def train_episode(self):
        start = time()
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()
        self.memory.set_initial_state(state)

        state_deque = deque(maxlen=self.num_sequences)
        action_deque = deque(maxlen=self.num_sequences-1)
        state_deque.append(state)

        while not done:
            if self.is_policy(state_deque):
                action = self.explore(state_deque, action_deque)
            else:
                action = 2 * np.random.rand(*self.action_shape) - 1
            next_state, reward, done, _ = self.env.step(action)
            self.steps += self.action_repeat
            episode_steps += self.action_repeat
            episode_reward += reward

            self.memory.append(action, reward, next_state, done)

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models()

            state_deque.append(next_state)
            action_deque.append(action)

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.training_log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)

        end = time()
        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}  '
              f'time: {end - start:<3.3f}')

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # First, update the latent model.
        images, actions, rewards, dones =\
            self.memory.sample(self.latent_batch_size)
        features = self.latent.encoder(images)
        latent_loss = self.calc_latent_loss(
            images, features, actions, rewards, dones)

        # Then, update policy and critic.
        images, actions, rewards, dones =\
            self.memory.sample(self.batch_size)

        # Don't update the latent model when updating policy and critic.
        with torch.no_grad():
            # Sample latent vectors from posterior dynamics.
            features = self.latent.encoder(images)
            (latents1, latents2), _ =\
                self.latent.sample_posterior(features, actions)
            latents = torch.cat([latents1, latents2], dim=-1)

            num_batches = features.size(0)
            feature_sequences = features.view(num_batches, -1)
            action_sequences = actions.view(num_batches, -1)
            trajectories = torch.cat(
                [feature_sequences, action_sequences], dim=-1)

        q1_loss, q2_loss = self.calc_critic_loss(
            latents, actions, trajectories, rewards, dones)
        policy_loss, entropies = self.calc_policy_loss(latents, trajectories)

        update_params(
            self.latent_optim, self.latent, latent_loss, self.grad_clip)
        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
        else:
            entropy_loss = 0.

        if self.learning_steps % self.learning_log_interval == 0:
            self.writer.add_scalar(
                'loss/latent', latent_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def calc_latent_loss(self, images, features, actions, rewards, dones):
        num_sequences = actions.size(1) + 1

        # Sample from posterior dynamics. (N, S, L*) samples, (S, N, L*) dists.
        (latent1_post_samples, latent2_post_samples),\
            (latent1_post_dists, latent2_post_dists) =\
            self.latent.sample_posterior(features, actions)

        # Sample from prior dynamics. (N, S, L*) samples, (S, N, L*) dists.
        (latent1_pri_samples, latent2_pri_samples),\
            (latent1_pri_dists, latent2_pri_dists) =\
            self.latent.sample_prior(actions)

        # Calculate KL divergence between priors and posteriors of z1.
        kld_loss = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)

        # Genarated observations: p(x(t) | z1(t), z2(t)).
        image_dists = self.latent.decoder(
            [latent1_post_samples, latent2_post_samples])
        # Log likelihood of x(t).
        log_likelihood_loss = image_dists.log_prob(images).mean(dim=0).sum()

        # Genarated rewards: p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        reward_dists = self.latent.reward_predictor([
            latent1_post_samples[:, :num_sequences-1],
            latent2_post_samples[:, :num_sequences-1],
            actions, latent1_post_samples[:, 1:num_sequences],
            latent2_post_samples[:, 1:num_sequences]])
        # Log likelihood of r(t) with masking where done = True.
        reward_log_likelihoods = reward_dists.log_prob(rewards) * (1.0-dones)
        reward_log_likelihood_loss = reward_log_likelihoods.mean(dim=0).sum()

        latent_loss =\
            kld_loss - log_likelihood_loss - reward_log_likelihood_loss

        if self.learning_steps % self.learning_log_interval == 0:
            images = images[:8, ...].detach().cpu()
            reconst_images = image_dists.loc[:8, ...].detach().cpu()

            t1_images = torch.cat(
                [images[:, 0], reconst_images[:, 0]], dim=-2)
            tlast_images = torch.cat(
                [images[:, -1], reconst_images[:, -1]], dim=-2)
            self.writer.add_images(
                'images/t=1/top_truth_bottom_reconst', t1_images,
                self.learning_steps)
            self.writer.add_images(
                'images/t=tau/top_truth_bottom_reconst', tlast_images,
                self.learning_steps)

            reconst_error = (
                images-reconst_images).pow(2).mean(dim=(0, 1)).sum().item()
            reward_reconst_error = ((
                rewards-reward_dists.loc).pow(2) * (1.0-dones)
                ).mean(dim=(0, 1)).sum().detach().item()
            self.writer.add_scalar(
                'stats/reconst_error', reconst_error, self.learning_steps)
            self.writer.add_scalar(
                'stats/reward_reconst_error', reward_reconst_error,
                self.learning_steps)

        return latent_loss

    def calc_critic_loss(self, latents, actions, trajectories, rewards, dones):
        # Q(z(t), a(t))
        curr_q1, curr_q2 = self.critic(latents[:, -2], actions[:, -1])

        # Q(z(t+1), a(t+1))
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(trajectories)
            next_q1, next_q2 = self.critic_target(latents[:, -1], next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q =\
            rewards[:, -1] + (1.0 - dones[:, -1]) * self.gamma * next_q

        # Critic losses are mean squared TD errors.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))

        if self.learning_steps % self.learning_log_interval == 0:
            mean_q1 = curr_q1.detach().mean().item()
            mean_q2 = curr_q2.detach().mean().item()
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)

        return q1_loss, q2_loss

    def calc_policy_loss(self, latents, trajectories):
        # Re-sample actions to calculate expectations of Q.
        sampled_actions, entropies, _ = self.policy.sample(trajectories)

        # Q(z(t+1), a(t+1))
        q1, q2 = self.critic(latents[:, -1], sampled_actions)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((- q - self.alpha * entropies))

        return policy_loss, entropies

    def calc_entropy_loss(self, entropies):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach())
        return entropy_loss

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False

            state_deque = deque(maxlen=self.num_sequences)
            action_deque = deque(maxlen=self.num_sequences-1)
            state_deque.append(state)

            while not done:
                if self.is_policy(state_deque):
                    action = self.explore(state_deque, action_deque)
                else:
                    action = 2 * np.random.rand(*self.action_shape) - 1
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state_deque.append(next_state)
                action_deque.append(action)

            returns[i] = episode_reward

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'steps: {self.steps:<5}  '
              f'reward: {mean_return:<5.1f} +/- {std_return:<5.1f}')
        print('-' * 60)

    def save_models(self):
        self.latent.save(os.path.join(self.model_dir, 'latent.pth'))
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        self.writer.close()
        self.env.close()
