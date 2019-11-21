import os
import argparse
import gym
from datetime import datetime
from dm_control import suite

from env import PixelObservationsDmControlWrapper,\
    PixelObservationsGymWrapper
from agent import SlacAgent


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', type=str, default='dm_control')
    parser.add_argument('--domain_name', type=str, default='cheetah')
    parser.add_argument('--task_name', type=str, default='run')
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'env_type': args.env_type,
        'num_steps': 3000000,
        'batch_size': 256,
        'latent_batch_size': 32,
        'num_sequences': 8,
        'action_repeat': args.action_repeat,
        'lr': 0.0003,
        'latent_lr': 0.0001,
        'feature_dim': 256,
        'latent1_dim': 32,
        'latent2_dim': 256,
        'hidden_units': [256, 256],
        'memory_size': 1e5,
        'gamma': 1.0,
        'target_update_interval': 1,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
        'leaky_slope': 0.2,
        'grad_clip': None,
        'updates_per_step': 1,
        'start_steps': 10000,
        'training_log_interval': 10,
        'learning_log_interval': 100,
        'eval_interval': 50000,
        'cuda': args.cuda,
        'seed': args.seed
    }

    if args.env_type == 'dm_control':
        env = suite.load(
            domain_name=args.domain_name, task_name=args.task_name)
        env = PixelObservationsDmControlWrapper(env, args.action_repeat)
        dir_name = f'{args.domain_name}-{args.task_name}'
    else:
        env = gym.make(args.env_id)
        env = PixelObservationsGymWrapper(env)
        dir_name = args.env_id

    log_dir = os.path.join(
        'logs', dir_name,
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}')

    agent = SlacAgent(env=env, log_dir=log_dir, **configs)
    agent.run()


if __name__ == '__main__':
    run()
