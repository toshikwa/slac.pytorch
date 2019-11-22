import os
from collections import deque
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from memory import LazyMemory
from network import LatentNetwork, GaussianPolicy, TwinnedQNetwork
from utils import create_feature_actions, calc_kl_divergence, grad_false,\
    hard_update, soft_update, update_params, RunningMeanStats


class SlacAgent:
    def __init__(self, env, log_dir, env_type='dm_control', num_steps=3000000,
                 initial_latent_steps=100000, batch_size=256,
                 latent_batch_size=32, num_sequences=8, lr=0.0003,
                 latent_lr=0.0001, feature_dim=256, latent1_dim=32,
                 latent2_dim=256, hidden_units=[256, 256], memory_size=1e5,
                 gamma=0.99, target_update_interval=1, tau=0.005,
                 entropy_tuning=True, ent_coef=0.2, leaky_slope=0.2,
                 grad_clip=None, updates_per_step=1, start_steps=10000,
                 training_log_interval=10, learning_log_interval=100,
                 eval_interval=50000, cuda=True, seed=0):

        self.env = env
        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.action_repeat = self.env.action_repeat

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.latent = LatentNetwork(
            self.observation_shape, self.action_shape, feature_dim,
            latent1_dim, latent2_dim, hidden_units, leaky_slope
            ).to(self.device)

        self.policy = GaussianPolicy(
            num_sequences * feature_dim
            + (num_sequences-1) * self.action_shape[0],
            self.action_shape[0], hidden_units).to(self.device)

        self.critic = TwinnedQNetwork(
            latent1_dim + latent2_dim, self.action_shape[0], hidden_units
            ).to(self.device)
        self.critic_target = TwinnedQNetwork(
            latent1_dim + latent2_dim, self.action_shape[0], hidden_units
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
        self.initial_latent_steps = initial_latent_steps
        self.num_sequences = num_sequences
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

    def reset_deque(self, state):
        state_deque = deque(maxlen=self.num_sequences)
        action_deque = deque(maxlen=self.num_sequences-1)

        for _ in range(self.num_sequences-1):
            state_deque.append(
                np.zeros(self.observation_shape, dtype=np.uint8))
            action_deque.append(
                np.zeros(self.action_shape, dtype=np.uint8))
        state_deque.append(state)

        return state_deque, action_deque

    def deque_to_batch(self, state_deque, action_deque):
        # Convert deques to batched tensor.
        state = np.array(state_deque, dtype=np.uint8)
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.0
        with torch.no_grad():
            feature = self.latent.encoder(state).view(1, -1)

        action = np.array(action_deque, dtype=np.float32)
        action = torch.FloatTensor(action).view(1, -1).to(self.device)
        feature_action = torch.cat([feature, action], dim=-1)
        return feature_action

    def explore(self, state_deque, action_deque):
        # Act with randomness
        feature_action = self.deque_to_batch(state_deque, action_deque)
        with torch.no_grad():
            action, _, _ = self.policy.sample(feature_action)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state_deque, action_deque):
        # Act without randomness
        feature_action = self.deque_to_batch(state_deque, action_deque)
        with torch.no_grad():
            _, _, action = self.policy.sample(feature_action)
        return action.cpu().numpy().reshape(-1)

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()
        self.memory.set_initial_state(state)
        state_deque, action_deque = self.reset_deque(state)

        while not done:
            if self.steps >= self.start_steps * self.action_repeat:
                action = self.explore(state_deque, action_deque)
            else:
                action = 2 * np.random.rand(*self.action_shape) - 1

            next_state, reward, done, _ = self.env.step(action)
            self.steps += self.action_repeat
            episode_steps += self.action_repeat
            episode_reward += reward

            self.memory.append(action, reward, next_state, done)

            if self.is_update():
                # First, train the latent model only.
                if self.learning_steps < self.initial_latent_steps:
                    print('-'*60)
                    print('Learning the latent model only...')
                    for _ in range(self.initial_latent_steps):
                        self.learning_steps += 1
                        self.learn_latent()
                    print('Finish learning the latent model.')
                    print('-'*60)

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

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}')

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # Update the latent model.
        self.learn_latent()
        # Update policy and critic.
        self.learn_sac()

    def learn_latent(self):
        images_seq, actions_seq, rewards_seq, dones_seq =\
            self.memory.sample_latent(self.latent_batch_size)
        latent_loss = self.calc_latent_loss(
            images_seq, actions_seq, rewards_seq, dones_seq)
        update_params(
            self.latent_optim, self.latent, latent_loss, self.grad_clip)

        if self.learning_steps % self.learning_log_interval == 0:
            self.writer.add_scalar(
                'loss/latent', latent_loss.detach().item(),
                self.learning_steps)

    def learn_sac(self):
        images_seq, actions_seq, rewards =\
            self.memory.sample_sac(self.batch_size)

        # NOTE: Don't update the encoder part of the policy here.
        with torch.no_grad():
            # f(1:t+1)
            features_seq = self.latent.encoder(images_seq)
            latent_samples, _ = self.latent.sample_posterior(
                features_seq, actions_seq)

        # z(t), z(t+1)
        latents_seq = torch.cat(latent_samples, dim=-1)
        latents = latents_seq[:, -2]
        next_latents = latents_seq[:, -1]
        # a(t)
        actions = actions_seq[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_actions, next_feature_actions =\
            create_feature_actions(features_seq, actions_seq)

        q1_loss, q2_loss = self.calc_critic_loss(
            latents, next_latents, actions, next_feature_actions,
            rewards)
        policy_loss, entropies = self.calc_policy_loss(
            latents, feature_actions)

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

    def calc_latent_loss(self, images_seq, actions_seq, rewards_seq,
                         dones_seq):
        features_seq = self.latent.encoder(images_seq)

        # Sample from posterior dynamics.
        (latent1_post_samples, latent2_post_samples),\
            (latent1_post_dists, latent2_post_dists) =\
            self.latent.sample_posterior(features_seq, actions_seq)
        # Sample from prior dynamics.
        (latent1_pri_samples, latent2_pri_samples),\
            (latent1_pri_dists, latent2_pri_dists) =\
            self.latent.sample_prior(actions_seq)

        # KL divergence loss.
        kld_loss = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)

        # Log likelihood loss of generated observations.
        images_seq_dists = self.latent.decoder(
            [latent1_post_samples, latent2_post_samples])
        log_likelihood_loss = images_seq_dists.log_prob(
            images_seq).mean(dim=0).sum()

        # Log likelihood loss of genarated rewards.
        rewards_seq_dists = self.latent.reward_predictor([
            latent1_post_samples[:, :-1],
            latent2_post_samples[:, :-1],
            actions_seq, latent1_post_samples[:, 1:],
            latent2_post_samples[:, 1:]])
        reward_log_likelihoods =\
            rewards_seq_dists.log_prob(rewards_seq) * (1.0 - dones_seq)
        reward_log_likelihood_loss = reward_log_likelihoods.mean(dim=0).sum()

        latent_loss =\
            kld_loss - log_likelihood_loss - reward_log_likelihood_loss

        if self.learning_steps % self.learning_log_interval == 0:
            reconst_error = (
                images_seq - images_seq_dists.loc
                ).pow(2).mean(dim=(0, 1)).sum().item()
            reward_reconst_error = ((
                rewards_seq - rewards_seq_dists.loc).pow(2) * (1.0 - dones_seq)
                ).mean(dim=(0, 1)).sum().detach().item()
            self.writer.add_scalar(
                'stats/reconst_error', reconst_error, self.learning_steps)
            self.writer.add_scalar(
                'stats/reward_reconst_error', reward_reconst_error,
                self.learning_steps)

        if self.learning_steps % (100 * self.learning_log_interval) == 0:
            gt_images = images_seq[0].detach().cpu()
            post_images = images_seq_dists.loc[0].detach().cpu()

            with torch.no_grad():
                pri_images = self.latent.decoder(
                        [latent1_pri_samples[:1], latent2_pri_samples[:1]]
                        ).loc[0].detach().cpu()
                cond_pri_samples, _ = self.latent.sample_prior(
                    actions_seq[:1], features_seq[:1, 0])
                cond_pri_images = self.latent.decoder(
                        cond_pri_samples).loc[0].detach().cpu()

            images = torch.cat(
                [gt_images, post_images, cond_pri_images, pri_images],
                dim=-2)

            # Visualize multiple of 8 images because each row contains 8
            # images at most.
            self.writer.add_images(
                'images/gt_posterior_cond-prior_prior',
                images[:(len(images)//8)*8], self.learning_steps)

        return latent_loss

    def calc_critic_loss(self, latents, next_latents, actions,
                         next_feature_actions, rewards):
        # Q(z(t), a(t))
        curr_q1, curr_q2 = self.critic(latents, actions)
        # E[Q(z(t+1), a(t+1)) + alpha * H(pi)]
        with torch.no_grad():
            next_actions, next_entropies, _ =\
                self.policy.sample(next_feature_actions)
            next_q1, next_q2 = self.critic_target(next_latents, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
        # r(t) + gamma * E[Q(z(t+1), a(t+1)) + alpha * H(pi)]
        target_q = rewards + self.gamma * next_q

        # Critic losses are mean squared TD errors.
        q1_loss = 0.5 * torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = 0.5 * torch.mean((curr_q2 - target_q).pow(2))

        if self.learning_steps % self.learning_log_interval == 0:
            mean_q1 = curr_q1.detach().mean().item()
            mean_q2 = curr_q2.detach().mean().item()
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)

        return q1_loss, q2_loss

    def calc_policy_loss(self, latents, feature_actions):
        # Re-sample actions to calculate expectations of Q.
        sampled_actions, entropies, _ = self.policy.sample(feature_actions)
        # E[Q(z(t), a(t))]
        q1, q2 = self.critic(latents, sampled_actions)
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
            state_deque, action_deque = self.reset_deque(state)

            while not done:
                action = self.explore(state_deque, action_deque)
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
        print(f'environment steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f} +/- {std_return:<5.1f}')
        print('-' * 60)

    def save_models(self):
        self.latent.encoder.save(os.path.join(self.model_dir, 'encoder.pth'))
        self.latent.save(os.path.join(self.model_dir, 'latent.pth'))
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        self.writer.close()
        self.env.close()
