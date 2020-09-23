from collections import deque

import numpy as np
import torch


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    """

    def __init__(self, frames):
        self._frames = frames
        self.dtype = frames[0].dtype
        self.out = None

    def _force(self):
        return np.array(self._frames, dtype=self.dtype)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class LazySequenceBuff:
    """
    Buffer to store the sequence of transitions efficiently.
    """

    def __init__(self, num_sequences=8):
        self.num_sequences = num_sequences
        self.reset()

    def reset(self):
        self._reset_episode = False
        self.state = deque(maxlen=self.num_sequences + 1)
        self.action = deque(maxlen=self.num_sequences)
        self.reward = deque(maxlen=self.num_sequences)
        self.done = deque(maxlen=self.num_sequences)

    def reset_episode(self, state):
        assert not self._reset_episode
        self._reset_episode = True
        self.state.append(state)

    def append(self, action, reward, done, next_state):
        assert self._reset_episode
        self.action.append(action)
        self.reward.append(np.array([reward], dtype=np.float32))
        self.done.append(np.array([done], dtype=np.bool))
        self.state.append(next_state)

    def get(self):
        state = LazyFrames(list(self.state))
        action = LazyFrames(list(self.action))
        reward = LazyFrames(list(self.reward))
        done = LazyFrames(list(self.done))
        return state, action, reward, done

    def __len__(self):
        return len(self.state)


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(self, buffer_size, num_sequences, state_shape, action_shape, device):
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device

        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.buff = LazySequenceBuff(num_sequences=self.num_sequences)

    def reset_episode(self, state):
        self.buff.reset_episode(state)

    def append(self, action, reward, done, next_state, episode_done):
        self.buff.append(action, reward, done, next_state)

        if len(self.buff) == self.num_sequences + 1:
            state, action, reward, done = self.buff.get()
            self._append(state, action, reward, done)

        if episode_done:
            self.buff.reset()

    def _append(self, state, action, reward, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)

        num_excess = len(self.reward) - self.buffer_size
        if num_excess > 0:
            del self.state[:num_excess]
            del self.action[:num_excess]
            del self.reward[:num_excess]
            del self.done[:num_excess]

    def sample_latent(self, batch_size):
        idxes = np.random.randint(low=0, high=len(self.reward), size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        action_ = np.empty((batch_size, self.num_sequences, *self.action_shape), dtype=np.float32)
        reward_ = np.empty((batch_size, self.num_sequences, 1), dtype=np.float32)
        done_ = np.empty((batch_size, self.num_sequences, 1), dtype=np.bool)

        # Convert LazeFrames into np.ndarray here.
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state[idx]
            action_[i, ...] = self.action[idx]
            reward_[i, ...] = self.reward[idx]
            done_[i, ...] = self.done[idx]

        state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        action_ = torch.tensor(action_, dtype=torch.float, device=self.device)
        reward_ = torch.tensor(reward_, dtype=torch.float, device=self.device)
        done_ = torch.tensor(done_, dtype=torch.float, device=self.device)
        return state_, action_, reward_, done_

    def sample_sac(self, batch_size):
        idxes = np.random.randint(low=0, high=len(self.reward), size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        action_ = np.empty((batch_size, self.num_sequences, *self.action_shape), dtype=np.float32)
        reward = np.empty((batch_size, 1), dtype=np.float32)

        # Convert LazeFrames into np.ndarray.
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state[idx]
            action_[i, ...] = self.action[idx]
            reward[i, ...] = self.reward[idx][-1]

        state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        action_ = torch.tensor(action_, dtype=torch.float, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        return state_, action_, reward

    def __len__(self):
        return len(self.reward)
