from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec


class PixelObservationsDmControlWrapper(wrappers.PyEnvironmentBaseWrapper):
    keys = ['state', 'pixels']

    def __init__(self, env, action_repeat, obs_type='pixels',
                 render_kwargs=None):
        super(PixelObservationsDmControlWrapper, self).__init__(env)
        assert obs_type in self.keys

        self._render_kwargs = dict(
            width=64,
            height=64,
            camera_id=0,
        )
        if render_kwargs is not None:
            self._render_kwargs.update(render_kwargs)

        if obs_type == 'state':
            self._observation_spec = self._env.observation_spec()
        elif obs_type == 'pixels':
            image_shape = (
                3, self._render_kwargs['height'],
                self._render_kwargs['width'])
            image_spec = array_spec.BoundedArraySpec(
                shape=image_shape, dtype=np.uint8, minimum=0, maximum=255)
            self._observation_spec = image_spec

        self.action_repeat = action_repeat
        self.obs_type = obs_type
        self.action_space = self.action_spec()
        self.observation_space = self.observation_spec()

    def observation_spec(self):
        return self._observation_spec

    def _modify_observation(self, observation):
        if self.obs_type == 'state':
            obs = observation
        elif self.obs_type == 'pixels':
            def get_physics(env):
                if hasattr(env, 'physics'):
                    return env.physics
                else:
                    return get_physics(env.wrapped_env())
            image = get_physics(self._env).render(**self._render_kwargs)
            obs = np.transpose(image, [2, 0, 1])
        return obs

    def _step(self, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            time_step = time_step._replace(
                observation=self._modify_observation(time_step.observation))
            reward += time_step.reward
            if time_step.step_type == ts.StepType.LAST:
                break

        return time_step.observation, reward,\
            time_step.step_type == ts.StepType.LAST, None

    def _reset(self):
        time_step = self._env.reset()
        time_step = time_step._replace(
            observation=self._modify_observation(time_step.observation))
        return time_step.observation

    def render(self, mode='rgb_array'):
        return self._env.render(mode=mode)
