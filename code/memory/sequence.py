import numpy as np
import torch


class Memory(dict):
    keys = ['state', 'action', 'reward', 'done']

    def __init__(self, capacity, observation_shape, action_shape,
                 use_flag=False):
        super(Memory, self).__init__()
        self.capacity = int(capacity)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.use_flag = use_flag
        self.reset()

    def reset(self):
        self._p = 0
        self._n = 0
        self['state'] = np.empty(
            (self.capacity, *self.observation_shape), dtype=np.uint8)
        self['action'] = np.empty(
            (self.capacity, *self.action_shape), dtype=np.float32)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)
        if self.use_flag:
            self['flag'] = np.zeros((self.capacity, ), dtype=np.bool)

    def append(self, state, action, reward, done):
        self['state'][self._p] = state
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def extend(self, states, actions, rewards, dones, flags):
        num_data = states.shape[0]

        if self._p + num_data <= self.capacity:
            target = slice(self._p, self._p+num_data)
            source = slice(0, num_data)
            self._insert(
                states, actions, rewards, dones, flags, target, source)

        else:
            target = slice(self._p, self.capacity)
            source = slice(0, self.capacity-self._p)
            self._insert(
                states, actions, rewards, dones, flags, target, source)

            target = slice(0, num_data-self.capacity-self._p)
            source = slice(self.capacity-self._p, num_data)
            self._insert(
                states, actions, rewards, dones, flags, target, source)

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, states, actions, rewards, dones, flags, target, source):
        self['state'][target] = states[source]
        self['action'][target] = actions[source]
        self['reward'][target] = rewards[source]
        self['done'][target] = dones[source]
        if self.use_flag:
            self['flag'][target] = flags[source]

    def get(self):
        states = self['state'][:self._n]
        actions = self['action'][:self._n]
        rewards = self['reward'][:self._n]
        dones = self['done'][:self._n]
        self.reset()

        return states, actions, rewards, dones

    def __len__(self):
        return self._n


class SequenceMemory:

    def __init__(self, capacity, num_sequences, observation_shape,
                 action_shape, device):
        self.capacity = int(capacity)
        self.num_sequences = int(num_sequences)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device

        self.sampling_flag = np.zeros((self.capacity, ), dtype=np.bool)
        self.episode_memory = Memory(
            capacity//10, observation_shape, action_shape)
        self.memory = Memory(
            capacity, observation_shape, action_shape, use_flag=True)

    def append(self, state, action, reward, done):
        self.episode_memory.append(state, action, reward, done)

        if done:
            if len(self.episode_memory) >= self.num_sequences:
                states, actions, rewards, dones = self.episode_memory.get()
                flags = np.ones((states.shape[0]), dtype=np.bool)
                flags[-self.num_sequences+2:] = False
                self.memory.extend(states, actions, rewards, dones, flags)
            else:
                self.episode_memory.reset()

    def sample(self, batch_size):
        ''' Sample batch of experiences.
        Output:
            states  : (N, S, *observation_shape) shaped tensor.
            actions : (N, S-1, *action_shape) shaped tensor.
            rewards : (N, S-1, 1) shaped tensor.
            dones   : (N, S-1, 1) shaped tensor.
        '''
        valid_indices = np.where(self.memory['flag'])[0]
        indices = np.random.choice(valid_indices, size=batch_size)

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
            if index + self.num_sequences <= self.capacity:
                target = slice(0, self.num_sequences)
                source = slice(index, index+self.num_sequences)
                states[i, target, ...] = self.memory['state'][source]

                target = slice(0, self.num_sequences-1)
                source = slice(index, index+self.num_sequences-1)
                actions[i, target, ...] = self.memory['action'][source]
                rewards[i, target, ...] = self.memory['reward'][source]
                dones[i, target, ...] = self.memory['done'][source]

            elif index + self.num_sequences + 1 <= self.capacity:
                states[i, 1:, ...] = self.memory[
                    'state'][index:index+self.num_sequences-1]
                states[i, 0, ...] = self.memory['state'][-1]

                target = slice(0, self.num_sequences-1)
                source = slice(index, index+self.num_sequences-1)
                actions[i, target, ...] = self.memory['action'][source]
                rewards[i, target, ...] = self.memory['reward'][source]
                dones[i, target, ...] = self.memory['done'][source]

            else:
                target = slice(0, self.capacity-index)
                source = slice(index, self.capacity)
                states[i, target, ...] = self.memory['state'][source]
                actions[i, target, ...] = self.memory['action'][source]
                rewards[i, target, ...] = self.memory['reward'][source]
                dones[i, target, ...] = self.memory['done'][source]

                source = slice(0, self.num_sequences-self.capacity+index)
                states[i, self.capacity-index:, ...] =\
                    self.memory['state'][source]
                actions[i, self.capacity-index:, ...] =\
                    self.memory['action'][source]
                rewards[i, self.capacity-index:, ...] =\
                    self.memory['reward'][source]
                dones[i, self.capacity-index:, ...] =\
                    self.memory['done'][source]

        states = torch.FloatTensor(states/255.0).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, dones

    def __len__(self):
        return len(self.memory)
