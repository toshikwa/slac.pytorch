from collections import deque
import numpy as np
import torch


class ScaledLazyFrames(object):
    def __init__(self, frames, axis=0, is_image=False):
        self._frames = frames
        self.axis = axis
        self.is_image = is_image

    def _force(self):
        if self.is_image:
            return np.stack(
                np.array(self._frames, dtype=np.float32)/255.0,
                axis=self.axis)
        else:
            return np.stack(
                np.array(self._frames, dtype=np.float32),
                axis=self.axis)

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
    keys = ['state', 'action', 'next_state']

    def __init__(self, num_sequences=8):
        super(SequenceBuff, self).__init__()
        self.num_sequences = int(num_sequences)
        self.memory = {
            key: deque(maxlen=self.num_sequences)
            for key in self.keys}

    def append(self, state, action, next_state):
        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['next_state'].append(next_state)

    def get(self):
        state = ScaledLazyFrames(list(self.memory['state']), 0, True)
        action = ScaledLazyFrames(list(self.memory['action']), 0)
        next_state = ScaledLazyFrames(list(self.memory['next_state']), 0, True)
        return state, action, next_state

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in SequenceBuff.')
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory['state'])


class Memory(dict):
    keys = ['state', 'action', 'reward', 'next_state', 'done']

    def __init__(self, capacity, num_sequences, device):
        super(Memory, self).__init__()
        self.capacity = int(capacity)
        self.num_sequences = int(num_sequences)
        self.device = device
        self.reset()

    def reset(self):
        self._p = 0
        self._n = 0
        for key in self.keys:
            self[key] = [None] * self.capacity
        self.buff = SequenceBuff(num_sequences=self.num_sequences)

    def append(self, state, action, reward, next_state, done):
        self.buff.append(state, action, next_state)

        if len(self.buff) == self.num_sequences:
            state, action, next_state = self.buff.get()
            self._append(state, action, reward, next_state, done)

        if done:
            self.buff.reset()

    def _append(self, state, action, reward, next_state, done):
        self['state'][self._p] = state
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['next_state'][self._p] = next_state
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((batch_size, 8, 3, 64, 64), dtype=np.float32)
        actions = np.empty((batch_size, 8, 6), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, 8, 3, 64, 64), dtype=np.float32)
        dones = np.empty((batch_size, 1), dtype=np.float32)

        for i, index in enumerate(indices):
            states[i, ...] = self['state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            next_states[i, ...] = self['next_state'][index]
            dones[i, ...] = self['done'][index]

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n
