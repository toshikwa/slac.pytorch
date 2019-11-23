import numpy as np
import gym
from dm_control import suite


class DmControlEnvForPytorch(gym.Env):
    keys = ['state', 'pixels']

    def __init__(self, domain_name, task_name, action_repeat=1,
                 obs_type='pixels', render_kwargs=None):
        super(DmControlEnvForPytorch, self).__init__()
        assert obs_type in self.keys
        self.env = suite.load(
            domain_name=domain_name, task_name=task_name)
        self.action_repeat = action_repeat
        self.obs_type = obs_type
        self.render_kwargs = dict(
            width=64,
            height=64,
            camera_id=0,
        )

        if render_kwargs is not None:
            self.render_kwargs.update(render_kwargs)

        if obs_type == 'state':
            obs_spec = self.env.observation_spec()
            obs_shape = (np.sum([a.shape[0] for a in obs_spec.values()]), )
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=obs_shape, dtype=np.float)
        elif obs_type == 'pixels':
            obs_shape = (
                3, self.render_kwargs['height'], self.render_kwargs['width'])
            self.observation_space = gym.spaces.Box(
                0, 255, shape=obs_shape, dtype=np.uint8)
        else:
            NotImplementedError

        action_spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(
            action_spec.minimum[0], action_spec.maximum[0],
            shape=action_spec.shape, dtype=action_spec.dtype)

    def _preprocess_obs(self, time_step):
        if self.obs_type == 'state':
            obs = np.concatenate([a for a in time_step.observation.values()])

        elif self.obs_type == 'pixels':
            def get_physics(env):
                if hasattr(env, 'physics'):
                    return env.physics
                else:
                    return get_physics(env.wrapped_env())
            image = get_physics(self.env).render(**self.render_kwargs)
            obs = np.transpose(image, [2, 0, 1])

        return obs

    def _step(self, action):
        time_step = self.env.step(action)
        reward = time_step.reward
        done = time_step.step_type == 2
        obs = self._preprocess_obs(time_step)

        return obs, reward, done, None

    def step(self, action):
        sum_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, done, _ = self._step(action)
            sum_reward += reward
            if done:
                break

        return obs, sum_reward, done, None

    def reset(self):
        return self._preprocess_obs(self.env.reset())

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def seed(self, seed):
        self.env.random = seed
