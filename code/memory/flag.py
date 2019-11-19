from collections import deque
import numpy as np
import torch


class FlagSequenceBuff:
    keys = ['state', 'action', 'reward', 'done']

    def __init__(self, action_shape, num_sequences=8):
        self.action_shape = action_shape
        self.num_sequences = int(num_sequences)
        self.memory = {
            'state': deque(maxlen=self.num_sequences),
            'action': deque(maxlen=self.num_sequences-1),
            'reward': deque(maxlen=self.num_sequences-1),
            'done': deque(maxlen=self.num_sequences-1)}

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def set_init_state(self, state):
        self.reset()
        self.memory['state'].append(state)

    def append(self, action, reward, next_state, done):
        self.memory['state'].append(next_state)
        self.memory['action'].append(action)
        self.memory['reward'].append(np.array([reward], dtype=np.float32))
        self.memory['done'].append(np.array([done], dtype=np.bool))

    def pop(self):
        assert len(self) == self.num_sequences
        state = self.memory['state'].popleft()
        action = self.memory['action'].popleft()
        reward = self.memory['reward'].popleft()
        done = self.memory['done'].popleft()
        flag = True

        return state, action, reward, done, flag

    def finish_episode(self):
        assert len(self) == self.num_sequences

        states = np.array(self.memory['state'], dtype=np.uint8)
        actions = np.zeros(
            (self.num_sequences, *self.action_shape), dtype=np.float32)
        rewards = np.zeros((self.num_sequences, 1), dtype=np.float32)
        dones = np.zeros((self.num_sequences, 1), dtype=np.bool)
        flags = np.zeros((self.num_sequences), dtype=np.bool)

        actions[:-1, ...] = self.memory['action']
        rewards[:-1, ...] = self.memory['reward']
        dones[:-1, ...] = self.memory['done']
        flags[0] = True

        self.reset()

        return states, actions, rewards, dones, flags

    def __len__(self):
        return len(self.memory['state'])


class FlagMemory(dict):
    keys = ['state', 'action', 'reward', 'done']

    def __init__(self, capacity, num_sequences, observation_shape,
                 action_shape, device):
        super(FlagMemory, self).__init__()
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

        self['state'] = np.empty(
            (self.capacity, *self.observation_shape), dtype=np.uint8)
        self['action'] = np.empty(
            (self.capacity, *self.action_shape), dtype=np.float32)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.bool)
        self['flag'] = np.zeros((self.capacity), dtype=np.bool)

        self.buff = FlagSequenceBuff(self.action_shape, self.num_sequences)

    def set_initial_state(self, state):
        self.buff.set_init_state(state)
        self.is_set_init = True

    def append(self, action, reward, next_state, done, episode_done=False):
        assert self.is_set_init is True

        self.buff.append(action, reward, next_state, done)

        if done or episode_done:
            states, actions, rewards, dones, flags = self.buff.finish_episode()
            self.extend(states, actions, rewards, dones, flags)

        elif len(self.buff) == self.num_sequences:
            state, action, reward, done, flag = self.buff.pop()
            self._append(state, action, reward, done, flag)

    def _append(self, state, action, reward, done, flag):
        self['state'][self._p] = state
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done
        self['flag'][self._p] = flag

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def extend(self, states, actions, rewards, dones, flags):
        num_data = states.shape[0]

        if self._p + num_data <= self.capacity:
            target = slice(self._p, self._p+num_data)
            source = slice(0, num_data)
            self._extend(
                states, actions, rewards, dones, flags, target, source)

        else:
            target = slice(self._p, self.capacity)
            source = slice(0, self.capacity-self._p)
            self._extend(
                states, actions, rewards, dones, flags, target, source)

            target = slice(0, num_data-self.capacity-self._p)
            source = slice(self.capacity-self._p, num_data)
            self._extend(
                states, actions, rewards, dones, flags, target, source)

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _extend(self, states, actions, rewards, dones, flags, target, source):
        self['state'][target] = states[source]
        self['action'][target] = actions[source]
        self['reward'][target] = rewards[source]
        self['done'][target] = dones[source]
        self['flag'][target] = flags[source]

    def sample_latent(self, batch_size):
        ''' Sample batch of experiences.
        Output:
            states  : (N, S, *observation_shape) shaped tensor.
            actions : (N, S-1, *action_shape) shaped tensor.
            rewards : (N, S-1, 1) shaped tensor.
            dones   : (N, S-1, 1) shaped tensor.
        '''
        valid_indices = np.where(self['flag'])[0]
        indices = np.random.choice(
            valid_indices, size=batch_size, replace=False)

        states = np.empty((
            batch_size, self.num_sequences, *self.observation_shape),
            dtype=np.uint8)
        actions = np.empty((
            batch_size, self.num_sequences-1, *self.action_shape),
            dtype=np.float32)
        rewards = np.empty((
            batch_size, self.num_sequences-1, 1), dtype=np.float32)
        dones = np.empty((
            batch_size, self.num_sequences-1, 1), dtype=np.bool)

        for i, index in enumerate(indices):
            if index + self.num_sequences + 1 <= self.capacity:
                s_start = index
                s_end = index + self.num_sequences

                if index + self.num_sequences <= self.capacity:
                    s_start = index
                    s_end = index + self.num_sequences
                    states[i, ...] = self['state'][s_start: s_end]
                else:
                    states[i, 1:, ...] = self['state'][index: s_end-1]
                    states[i, 0, ...] = self['state'][s_end-1]

                actions[i, ...] = self['action'][s_start: s_end-1]
                rewards[i, ...] = self['reward'][s_start: s_end-1]
                dones[i, ...] = self['done'][s_start: s_end-1]

            else:
                s_start = index
                s_end = self.capacity
                t_end = self.capacity - index
                states[i, : t_end, ...] = self['state'][s_start: s_end]
                actions[i, : t_end, ...] = self['action'][s_start: s_end]
                rewards[i, : t_end, ...] = self['reward'][s_start: s_end]
                dones[i, : t_end, ...] = self['done'][s_start: s_end]

                s_end = self.num_sequences - t_end
                t_start = t_end

                states[i, t_start:, ...] = self['state'][: s_end]
                actions[i, t_start:, ...] = self['action'][: s_end-1]
                rewards[i, t_start:, ...] = self['reward'][: s_end-1]
                dones[i, t_start:, ...] = self['done'][: s_end-1]

        states = torch.CharTensor(states).to(self.device).float() / 255.0
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device).float()

        return states, actions, rewards, dones

    def sample_sac(self, batch_size):
        ''' Sample batch of experiences.
        Output:
            states  : (N, S, *observation_shape) shaped tensor.
            actions : (N, S-1, *action_shape) shaped tensor.
            rewards : (N, 1) shaped tensor.
            dones   : (N,1) shaped tensor.
        '''
        valid_indices = np.where(self['flag'])[0]
        indices = np.random.choice(
            valid_indices, size=batch_size, replace=False)

        states = np.empty((
            batch_size, self.num_sequences, *self.observation_shape),
            dtype=np.uint8)
        actions = np.empty((
            batch_size, self.num_sequences-1, *self.action_shape),
            dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        dones = np.empty((batch_size, 1), dtype=np.bool)

        for i, index in enumerate(indices):
            if index + self.num_sequences + 1 <= self.capacity:
                s_start = index
                s_end = index + self.num_sequences

                if index + self.num_sequences <= self.capacity:
                    s_start = index
                    s_end = index + self.num_sequences
                    states[i, ...] = self['state'][s_start: s_end]
                else:
                    states[i, 1:, ...] = self['state'][index: s_end-1]
                    states[i, 0, ...] = self['state'][s_end-1]

                actions[i, ...] = self['action'][s_start: s_end-1]
                rewards[i, ...] = self['reward'][s_end-2]
                dones[i, ...] = self['done'][s_end-2]

            else:
                s_start = index
                s_end = self.capacity
                t_end = self.capacity - index
                states[i, : t_end, ...] = self['state'][s_start: s_end]
                actions[i, : t_end, ...] = self['action'][s_start: s_end]

                s_end = self.num_sequences - t_end
                t_start = t_end

                states[i, t_start:, ...] = self['state'][: s_end]
                actions[i, t_start:, ...] = self['action'][: s_end-1]
                rewards[i, t_start:, ...] = self['reward'][s_end-2]
                dones[i, t_start:, ...] = self['done'][s_end-2]

        states = torch.CharTensor(states).to(self.device).float() / 255.0
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device).float()

        return states, actions, rewards, dones

    def __len__(self):
        return self._n
