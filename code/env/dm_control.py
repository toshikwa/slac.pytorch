from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec


class PixelObservationsDmControlWrapper(wrappers.PyEnvironmentBaseWrapper):
    keys = ['state', 'pixels']

    def __init__(self, env, action_repeat, render_kwargs=None):
        super(PixelObservationsDmControlWrapper, self).__init__(env)
        self._render_kwargs = dict(
            width=64,
            height=64,
            camera_id=0,
        )
        if render_kwargs is not None:
            self._render_kwargs.update(render_kwargs)

        observation_spec = collections.OrderedDict()
        for observation_name in self.keys:
            if observation_name == 'state':
                observation_spec['state'] = self._env.observation_spec()
            elif observation_name == 'pixels':
                image_shape = (
                    3, self._render_kwargs['height'],
                    self._render_kwargs['width'])
                image_spec = array_spec.BoundedArraySpec(
                    shape=image_shape, dtype=np.uint8, minimum=0, maximum=255)
                observation_spec['pixels'] = image_spec
        self._observation_spec = observation_spec

        self.action_repeat = action_repeat

    def observation_spec(self):
        return self._observation_spec

    def _modify_observation(self, observation):
        observations = collections.OrderedDict()
        for observation_name in self.keys:
            if observation_name == 'state':
                observations['state'] = observation
            elif observation_name == 'pixels':
                def get_physics(env):
                    if hasattr(env, 'physics'):
                        return env.physics
                    else:
                        return get_physics(env.wrapped_env())
                image = get_physics(self._env).render(**self._render_kwargs)
                observations['pixels'] = np.transpose(image, [2, 0, 1])

        return observations['pixels']

    def _gym_output(self, time_step):
        obs = time_step.observation
        reward = time_step.reward
        done = time_step.step_type == ts.StepType.LAST

        return obs, reward, done, None

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


# class ActionRepeatDmControlWrapper(PixelObservationsDmControlWrapper):

#     def __init__(self, env, action_repeat, render_kwargs=None):
#         super(ActionRepeatDmControlWrapper, self).__init__(env, render_kwargs)
#         self.action_repeat = action_repeat

#     def _step(self, action):
#         total_reward = 0.0
#         for _ in range(self.action_repeat):
#             obs, reward, done, _ = super(
#                 ActionRepeatDmControlWrapper, self)._step(action)
#             total_reward += reward
#             if done:
#                 break
#         return obs, total_reward, done, None

#     def _reset(self):
#         return self._env.reset()
