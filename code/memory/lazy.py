from collections import deque
import numpy as np
import torch


class LazyFrames:
    ''' LazyFrames memory-efficiently stores stacked data. '''

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
    keys = ['state', 'action', 'reward', 'done']

    def __init__(self, num_sequences=8):
        self.num_sequences = int(num_sequences)

    def reset(self):
        self.memory = {
            'state': deque(maxlen=self.num_sequences + 1),
            'action': deque(maxlen=self.num_sequences),
            'reward': deque(maxlen=self.num_sequences),
            'done': deque(maxlen=self.num_sequences)}

    def set_init_state(self, state):
        self.reset()
        self.memory['state'].append(state)

    def append(self, action, reward, next_state, done):
        self.memory['state'].append(next_state)
        self.memory['action'].append(action)
        self.memory['reward'].append(np.array([reward], dtype=np.float32))
        self.memory['done'].append(np.array([done], dtype=np.bool))

    def get(self):
        # It's memory-efficient, but slow.
        states = LazyFrames(list(self.memory['state']))
        actions = LazyFrames(list(self.memory['action']))
        rewards = LazyFrames(list(self.memory['reward']))
        dones = LazyFrames(list(self.memory['done']))

        return states, actions, rewards, dones

    def __len__(self):
        return len(self.memory['state'])


class LazyMemory(dict):
    ''' LazyMemory is memory-efficient but time-inefficient. '''
    keys = ['state', 'action', 'reward', 'done']

    def __init__(self, capacity, num_sequences, observation_shape,
                 action_shape, device):
        super(LazyMemory, self).__init__()
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
        self.buff = LazySequenceBuff(num_sequences=self.num_sequences)

    def set_initial_state(self, state):
        self.buff.set_init_state(state)
        self.is_set_init = True

    def append(self, action, reward, next_state, done, episode_done=False):
        assert self.is_set_init is True

        self.buff.append(action, reward, next_state, done)

        if len(self.buff) == self.num_sequences + 1:
            states, actions, rewards, dones = self.buff.get()
            self._append(states, actions, rewards, dones)

        if done or episode_done:
            self.buff.reset()

    def _append(self, state, action, reward, done):
        self['state'][self._p] = state
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample_latent(self, batch_size):
        '''
        Returns:
            state : (N, S+1, *observation_shape) shaped tensor.
            action: (N, S, *action_shape) shaped tensor.
            reward: (N, S, 1) shaped tensor.
            done  : (N, S, 1) shaped tensor.
        '''
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((
            batch_size, self.num_sequences+1, *self.observation_shape),
            dtype=np.uint8)
        actions = np.empty((
            batch_size, self.num_sequences, *self.action_shape),
            dtype=np.float32)
        rewards = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.float32)
        dones = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.bool)

        for i, index in enumerate(indices):
            # Convert LazeFrames into np.ndarray here.
            states[i, ...] = self['state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            dones[i, ...] = self['done'][index]

        states = torch.ByteTensor(states).to(self.device).float() / 255.0
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device).float()

        return states, actions, rewards, dones

    def sample_sac(self, batch_size):
        '''
        Returns:
            state : (N, S+1, *observation_shape) shaped tensor.
            action: (N, S, *action_shape) shaped tensor.
            reward: (N, 1) shaped tensor.
            done  : (N, 1) shaped tensor.
        '''
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((
            batch_size, self.num_sequences+1, *self.observation_shape),
            dtype=np.uint8)
        actions = np.empty((
            batch_size, self.num_sequences, *self.action_shape),
            dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        dones = np.empty((batch_size, 1), dtype=np.bool)

        for i, index in enumerate(indices):
            # Convert LazeFrames into np.ndarray here.
            states[i, ...] = self['state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index][-1]
            dones[i, ...] = self['done'][index][-1]

        states = torch.ByteTensor(states).to(self.device).float() / 255.0
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device).float()

        return states, actions, rewards, dones

    def __len__(self):
        return self._n
