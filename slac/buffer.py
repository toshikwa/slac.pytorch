from collections import deque

import numpy as np
import torch


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def __array__(self, dtype):
        return np.array(self._frames, dtype=dtype)

    def __len__(self):
        return len(self._frames)


class SequenceBuff:
    """
    Buffer to store a sequence of data efficiently.
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
        self.reward.append([reward])
        self.done.append([done])
        self.state.append(next_state)

    def get(self, device):
        state = LazyFrames(self.state)
        action = np.array(self.action, dtype=np.float32)
        reward = np.array(self.reward, dtype=np.float32)
        done = np.array(self.done, dtype=np.float32)
        return state, action, reward, done

    def __len__(self):
        return len(self.done)


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(self, buffer_size, num_sequences, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device

        # Store the sequence of images as a list of LazyFrames on CPU. It can store images with 9 times less memory.
        self.state = [None] * buffer_size
        # Store other data on GPU to reduce workloads.
        self.action = torch.empty(buffer_size, num_sequences, *action_shape, device=device)
        self.reward = torch.empty(buffer_size, num_sequences, 1, device=device)
        self.done = torch.empty(buffer_size, num_sequences, 1, device=device)
        # Buffer to store a sequence of trajectories.
        self.buff = SequenceBuff(num_sequences=self.num_sequences)

    def reset_episode(self, state):
        """
        Reset the buffer and set the initial observation. This has to be done before every episode starts.
        """
        self.buff.reset_episode(state)

    def append(self, action, reward, done, next_state, episode_done):
        """
        Store trajectory in the buffer. If the buffer is full, the sequence of trajectories is stored in replay buffer.
        Please pass 'masked' and 'true' done so that we can assert if the start/end of an episode is handled properly.
        """
        self.buff.append(action, reward, done, next_state)

        if len(self.buff) == self.num_sequences:
            state, action, reward, done = self.buff.get(self.device)
            self._append(state, action, reward, done)

        if episode_done:
            self.buff.reset()

    def _append(self, state, action, reward, done):
        self.state[self._p] = state
        self.action[self._p].copy_(torch.from_numpy(action))
        self.reward[self._p].copy_(torch.from_numpy(reward))
        self.done[self._p].copy_(torch.from_numpy(done))

        self._n = min(self._n + 1, self.buffer_size)
        self._p = (self._p + 1) % self.buffer_size

    def sample_latent(self, batch_size):
        """
        Sample trajectories for updating latent variable model.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state[idx]
        state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        return state_, self.action[idxes], self.reward[idxes], self.done[idxes]

    def sample_sac(self, batch_size):
        """
        Sample trajectories for updating SAC.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state[idx]
        state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        return state_, self.action[idxes], self.reward[idxes, -1]

    def __len__(self):
        return self._n
