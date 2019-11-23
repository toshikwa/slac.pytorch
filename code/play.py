import os
import argparse
from collections import deque
import numpy as np
import torch

from env import DmControlEnvForPytorch, GymEnvForPyTorch
from network import EvalPolicy
from utils import grad_false


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', type=str, default='dm_control')
    parser.add_argument('--domain_name', type=str, default='cheetah')
    parser.add_argument('--task_name', type=str, default='run')
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--log_name', type=str, default='slac-seed0-datetime')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.env_type == 'dm_control':
        env = DmControlEnvForPytorch(
            args.domain_name, args.task_name, args.action_repeat)
        dir_name = f'{args.domain_name}-{args.task_name}'
    else:
        env = GymEnvForPyTorch(args.env_id, args.action_repeat)
        dir_name = args.env_id

    log_dir = os.path.join('logs', args.env_type, dir_name, args.log_name)
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    policy = EvalPolicy(
        env.observation_space.shape,
        env.action_space.shape).to(device).eval()

    policy.load_weights(os.path.join(log_dir, 'model'))
    grad_false(policy)

    def reset_deque(state):
        num_sequences = 8
        state_deque = deque(maxlen=num_sequences)
        action_deque = deque(maxlen=num_sequences-1)

        for _ in range(num_sequences-1):
            state_deque.append(
                np.zeros(env.observation_space.shape, dtype=np.uint8))
            action_deque.append(
                np.zeros(env.action_space.shape, dtype=np.uint8))
        state_deque.append(state)

        return state_deque, action_deque

    def exploit(state_deque, action_deque):
        state = np.array(state_deque, dtype=np.uint8)
        state = torch.ByteTensor(state).unsqueeze(0).to(device).float() / 255.0
        action = np.array(action_deque, dtype=np.float32)
        action = torch.FloatTensor(action).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, action = policy.sample(state, action)
        return action.cpu().numpy().reshape(-1)

    state = env.reset()
    state_deque, action_deque = reset_deque(state)

    episode_reward = 0.
    done = False
    while not done:
        action = exploit(state_deque, action_deque)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state


if __name__ == '__main__':
    run()
