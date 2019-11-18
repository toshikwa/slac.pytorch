from collections import deque
import numpy as np
import torch


class LazyFrames(object):
    def __init__(self, frames, is_image=False):
        self._frames = frames
        self.is_image = is_image

    def _force(self):
        if self.is_image:
            return np.stack(
                np.array(self._frames, dtype=np.float32)/255.0,
                axis=0)
        else:
            return np.stack(
                np.array(self._frames, dtype=np.float32),
                axis=0)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class Frames(object):
    def __init__(self, frames, is_image=False):
        self._frames = frames
        self.is_image = is_image

    def _force(self):
        if self.is_image:
            return np.array(self._frames, dtype=np.float32) / 255.0
        else:
            return self._frames

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class SequenceBuff:
    keys = ['state', 'action', 'reward', 'done']

    def __init__(self, num_sequences=8):
        super(SequenceBuff, self).__init__()
        self.num_sequences = int(num_sequences)

    def reset(self):
        self.memory = {
            'state': deque(maxlen=self.num_sequences),
            'action': deque(maxlen=self.num_sequences-1),
            'reward': deque(maxlen=self.num_sequences-1),
            'done': deque(maxlen=self.num_sequences-1)}

    def set_init_state(self, state):
        self.reset()
        self.memory['state'].append(state)

    def append(self, action, reward, next_state, done):
        self.memory['state'].append(next_state)
        self.memory['action'].append(action)
        self.memory['reward'].append(np.array([reward], dtype=np.float32))
        self.memory['done'].append(np.array([done], dtype=np.float32))

    def get(self):
        state = LazyFrames(list(self.memory['state']), True)
        action = LazyFrames(list(self.memory['action']))
        reward = LazyFrames(list(self.memory['reward']))
        done = LazyFrames(list(self.memory['done']))

        return state, action, reward, done

    def __len__(self):
        return len(self.memory['state'])


class LazyMemory(dict):
    keys = ['state', 'action', 'reward', 'done']

    def __init__(self, capacity, num_sequences, observation_shape,
                 action_shape, device):
        super(Memory, self).__init__()
        self.capacity = int(capacity)
        self.num_sequences = int(num_sequences)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device
        self.reset()

    def reset(self):
        self.is_set_init = False
        self._p = 0
        self._n = 0
        for key in self.keys:
            self[key] = [None] * self.capacity
        self.buff = SequenceBuff(num_sequences=self.num_sequences)

    def set_initial_state(self, state):
        self.buff.set_init_state(state)
        self.is_set_init = True

    def append(self, action, reward, next_state, done, episode_done=False):
        assert self.is_set_init is True

        self.buff.append(action, reward, next_state, done)

        if len(self.buff) == self.num_sequences:
            state, action, reward, done = self.buff.get()
            self._append(state, action, reward, done)

        if done or episode_done:
            self.buff.reset()

    def _append(self, state, action, reward, done):
        self['state'][self._p] = state
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((
            batch_size, self.num_sequences, *self.observation_shape),
            dtype=np.float32)
        actions = np.empty((
            batch_size, self.num_sequences-1, *self.action_shape),
            dtype=np.float32)
        rewards = np.empty((
            batch_size, self.num_sequences-1, 1), dtype=np.float32)
        dones = np.empty((
            batch_size, self.num_sequences-1, 1), dtype=np.float32)

        for i, index in enumerate(indices):
            states[i, ...] = np.array(self['state'][index])
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            dones[i, ...] = self['done'][index]

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, dones

    def __len__(self):
        return self._n
