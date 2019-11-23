import os
import argparse
from datetime import datetime

from env import DmControlEnvForPytorch, GymEnvForPyTorch
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

    # Configs which are constant across all tasks.
    configs = {
        'env_type': args.env_type,
        'num_steps': 3000000,
        'initial_latent_steps': 100000,
        'batch_size': 256,
        'latent_batch_size': 32,
        'num_sequences': 8,
        'lr': 0.0003,
        'latent_lr': 0.0001,
        'feature_dim': 256,
        'latent1_dim': 32,
        'latent2_dim': 256,
        'hidden_units': [256, 256],
        'memory_size': 100000,
        'gamma': 0.99,
        'target_update_interval': 1,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
        'leaky_slope': 0.2,
        'grad_clip': None,
        'updates_per_step': 1,
        'start_steps': 10000,
        'training_log_interval': 4,
        'learning_log_interval': 100,
        'eval_interval': 10000,
        'cuda': args.cuda,
        'seed': args.seed
    }

    if args.env_type == 'dm_control':
        env = DmControlEnvForPytorch(
            args.domain_name, args.task_name, args.action_repeat)
        dir_name = f'{args.domain_name}-{args.task_name}'
    else:
        env = GymEnvForPyTorch(args.env_id, args.action_repeat)
        dir_name = args.env_id

    log_dir = os.path.join(
        'logs', args.env_type, dir_name,
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}')

    agent = SlacAgent(env=env, log_dir=log_dir, **configs)
    agent.run()


if __name__ == '__main__':
    run()
