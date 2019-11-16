import os
from collections import deque
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from memory.sequential import Memory
from network.policy import GaussianPolicy
from network.q_func import TwinnedQNetwork
from network.latent import LatentNetwork
from utils import grad_false, hard_update, soft_update, update_params,\
    RunningMeanStats


class SlacAgent:
    def __init__(self, env, log_dir, env_type='dm_control', num_steps=3000000,
                 batch_size=256, num_sequences=8, lr=0.0003, feature_dim=256,
                 latent1_dim=32, latent2_dim=256,
                 hidden_units=[256, 256], memory_size=1e5, gamma=0.99,
                 tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, cuda=True, seed=0):
        self.env = env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        if env_type == 'dm_control':
            self.action_shape = self.env.action_spec().shape
            self.observation_shape =\
                self.env.observation_spec()['pixels'].shape
        elif env_type == 'gym':
            self.action_shape = self.env.action_space.shape
            self.observation_shape = self.env.observation_space['pixels'].shape

        self.policy = GaussianPolicy(
            num_sequences * self.observation_shape[0]
            + (num_sequences-1) * self.action_shape[0],
            self.action_shape[0], hidden_units).to(self.device)

        self.critic = TwinnedQNetwork(
            latent1_dim+latent2_dim, self.action_shape[0], hidden_units
            ).to(self.device)
        self.critic_target = TwinnedQNetwork(
            latent1_dim+latent2_dim, self.action_shape[0], hidden_units
            ).to(self.device).eval()

        self.latent = LatentNetwork(
            self.observation_shape, self.action_shape, feature_dim,
            latent1_dim, latent2_dim)

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)
        self.latent_optim = Adam(self.latent.parameters(), lr=lr)

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
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        self.memory = Memory(memory_size, num_sequences, self.device)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_sequences = num_sequences
        self.num_steps = num_steps
        self.tau = tau
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps

    def act(self, states, actions):
        if self.start_steps > self.steps:
            action = 2 * np.random.rand(*self.action_shape) - 1
        else:
            action = self.explore(states, actions)
        return action

    def to_batch(self, states, actions):
        with torch.no_grad():
            features = self.latent.encoder(states)
        num_batches = features.size(0)
        sequence_feature = features.view(num_batches, -1)
        sequence_action = (actions[:, :-1]).view(num_batches, -1)
        experiences = torch.cat([sequence_feature, sequence_action], dim=-1)
        return experiences

    def explore(self, states, actions):
        # act with randomness
        experiences = self.to_batch(states, actions)
        with torch.no_grad():
            action, _, _ = self.policy.sample(experiences)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, states, actions):
        # act without randomness
        experiences = self.to_batch(states, actions)
        with torch.no_grad():
            _, _, action = self.policy.sample(experiences)
        return action

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        obs = self.env.reset()

        state_deque = deque(maxlen=self.num_sequences)
        action_deque = deque(maxlen=self.num_sequences-1)
        while len(state_deque) != self.num_sequences:
            state_deque.append(obs['pixels'])
        while len(action_deque) != self.num_sequences-1:
            action_deque.append(2*np.random.rand(*self.action_shape)-1)

        while not done:
            state = np.array(state_deque, np.float32)
            action = np.array(action_deque, np.float32)

            next_action = self.act(state, action)
            next_state, reward, done, _ = self.env.step(next_action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            self.memory.append(
                state, next_action, reward, next_state, masked_done)

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models()

            state_deque.append(next_state)
            action_deque.append(next_action)

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}')

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        batch = self.memory.sample(self.batch_size)

        q1_loss, q2_loss = self.calc_critic_loss(*batch)
        policy_loss, entropies = self.calc_policy_loss(*batch)

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

        if self.learning_steps % self.log_interval == 0:
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

    def calc_critic_loss(self, states, actions, rewards, next_states,
                         dones):
        # current Q
        (latent1, latent2), _ =\
            self.latent.sample_posterior(states, actions)
        latent = torch.cat([latent1, latent2], dim=-1)
        curr_q1, curr_q2 = self.critic(latent[:, -2], actions[:, -2])

        # target Q
        with torch.no_grad():
            next_exps = self.to_batch(next_states, actions)
            next_actions, next_entropies, _ = self.policy.sample(next_exps)
            next_q1, next_q2 = self.critic_target(latent[:, -1], next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))
        return q1_loss, q2_loss

    def calc_policy_loss(self, states, actions, rewards, next_states,
                         dones):
        # re-sample actions to calculate expectations of Q.
        exps = self.to_batch(states, actions)
        sampled_action, entropy, _ = self.policy.sample(exps)

        # expectations of Q with clipped double Q technique
        (latent1, latent2), _ =\
            self.latent.sample_posterior(states, actions)
        latent = torch.cat([latent1, latent2], dim=-1)
        q1, q2 = self.critic(latent[-1], sampled_action)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = torch.mean((- q - self.alpha * entropy))

        return policy_loss, entropy

    def calc_entropy_loss(self, entropy):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach())
        return entropy_loss
