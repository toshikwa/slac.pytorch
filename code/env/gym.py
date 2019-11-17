import collections
from gym import Wrapper, spaces
import numpy as np


class RenderGymWrapper(Wrapper):

    def __init__(self, env, render_kwargs=None):
        super(RenderGymWrapper, self).__init__(env)
        self._render_kwargs = dict(
            width=64,
            height=64,
            depth=False,
            camera_name='track',
        )
        if render_kwargs is not None:
            self._render_kwargs.update(render_kwargs)

    @property
    def sim(self):
        return self._env.sim

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self._env.sim.render(**self._render_kwargs)[::-1, :, :]
        else:
            return self._env.render(mode=mode)


class PixelObservationsGymWrapper(Wrapper):
    keys = ['state', 'pixels']

    def __init__(self, env, obs_type='pixels', render_kwargs=None):
        super(PixelObservationsGymWrapper, self).__init__(env)
        assert obs_type in self.keys

        self._render_kwargs = dict(
            width=64,
            height=64,
            depth=False,
            camera_name='track',
        )

        if render_kwargs is not None:
            self._render_kwargs.update(render_kwargs)

        if obs_type == 'state':
            self.observation_space = self.env.observation_space
        elif obs_type == 'pixels':
            image_shape = (
                3, self._render_kwargs['height'], self._render_kwargs['width'])
            image_space = spaces.Box(
                0, 255, shape=image_shape, dtype=np.uint8)
            self.observation_space = image_space

        self.obe_type = obs_type

    def _modify_observation(self, observation):
        if self.obe_type == 'state':
            obs = observation
        elif self.obe_type == 'pixels':
            image = self.env.sim.render(**self._render_kwargs)[::-1, :, :]
            obs = np.transpose(image, (2, 0, 1))

        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self._modify_observation(observation)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        return self._modify_observation(observation)

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
