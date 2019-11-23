import gym
import numpy as np


class GymEnvForPyTorch(gym.Env):
    keys = ['state', 'pixels']

    def __init__(self, env_id, action_repeat=1, obs_type='pixels',
                 render_kwargs=None):
        assert obs_type in self.keys
        self.env = gym.make(env_id)
        self.action_repeat = action_repeat
        self.obs_type = obs_type

        self.render_kwargs = dict(
            width=64,
            height=64,
            depth=False,
            camera_name='track',
        )

        if render_kwargs is not None:
            self.render_kwargs.update(render_kwargs)

        if obs_type == 'state':
            self.observation_space = self.env.observation_space
        elif obs_type == 'pixels':
            obs_shape = (
                3, self.render_kwargs['height'], self.render_kwargs['width'])
            self.observation_space = gym.spaces.Box(
                0, 255, shape=obs_shape, dtype=np.uint8)
        else:
            NotImplementedError

        self.action_space = self.env.action_space

    def _preprocess_obs(self, obs):
        if self.obs_type == 'pixels':
            image = self.env.sim.render(**self.render_kwargs)[::-1, :, :]
            obs = np.transpose(image, [2, 0, 1])
        return obs

    def step(self, action):
        sum_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, done, _ = self.env.step(action)
            sum_reward += reward
            if done:
                break
        return self._preprocess_obs(obs), sum_reward, done, None

    def reset(self):
        obs = self.env.reset()
        return self._preprocess_obs(obs)

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def seed(self, seed):
        self.env.seed(seed)
