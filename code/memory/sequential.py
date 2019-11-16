from collections import deque
import random
import numpy as np
import torch


class ScaledLazyFrames(object):
    def __init__(self, frames, axis=0):
        self._frames = frames
        self.axis = axis

    def _force(self):
        return np.stack(
            np.array(self._frames, dtype=np.float32)/255.0,
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
        state = ScaledLazyFrames(list(self.memory['state']), 0)
        action = ScaledLazyFrames(list(self.memory['action']), 0)
        next_state = ScaledLazyFrames(list(self.memory['next_state']), 0)
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


class Memory:
    keys = ['state', 'action', 'reward', 'next_state', 'done']

    def __init__(self, capacity, num_sequences, device):
        super(Memory, self).__init__()
        self.capacity = int(capacity)
        self.num_sequences = int(num_sequences)
        self.device = device
        self.reset()

    def reset(self):
        self._p = 0
        self.memory = []
        self.buff = SequenceBuff(num_sequences=self.num_sequences)

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self.buff.append(state, action, next_state)

        if len(self.buff) == self.num_sequences:
            state, action, next_state = self.buff.get()
            self._append(state, action, reward, next_state, done)

        if episode_done or done:
            self.buff.reset()

    def _append(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self._p] = (state, action, reward, next_state, done)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones =\
            map(np.stack, zip(*batch))

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
