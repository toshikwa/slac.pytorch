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

    def __init__(self, env, render_kwargs=None):
        super(PixelObservationsGymWrapper, self).__init__(env)
        self._render_kwargs = dict(
            width=64,
            height=64,
            depth=False,
            camera_name='track',
        )
        if render_kwargs is not None:
            self._render_kwargs.update(render_kwargs)

        observation_spaces = collections.OrderedDict()
        for observation_name in self.keys:
            if observation_name == 'state':
                observation_spaces['state'] = self.env.observation_space
            elif observation_name == 'pixels':
                image_shape = (
                    self._render_kwargs['height'],
                    self._render_kwargs['width'], 3)
                image_space = spaces.Box(
                    0, 255, shape=image_shape, dtype=np.uint8)
                observation_spaces['pixels'] = image_space

        self.observation_space = spaces.Dict(observation_spaces)

    def _modify_observation(self, observation):
        observations = collections.OrderedDict()
        for observation_name in self.keys:
            if observation_name == 'state':
                observations['state'] = observation
            elif observation_name == 'pixels':
                image = self.env.sim.render(**self._render_kwargs)[::-1, :, :]
                observations['pixels'] = image

        return observations

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self._modify_observation(observation)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        return self._modify_observation(observation)

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
