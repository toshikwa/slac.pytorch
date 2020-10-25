import argparse
import os
from datetime import datetime

import torch

from slac.algo import SlacAlgorithm
from slac.env import make_dmc
from slac.trainer import Trainer


def main(args):
    env = make_dmc(
        domain_name=args.domain_name,
        task_name=args.task_name,
        action_repeat=args.action_repeat,
        image_size=64,
    )
    env_test = make_dmc(
        domain_name=args.domain_name,
        task_name=args.task_name,
        action_repeat=args.action_repeat,
        image_size=64,
    )

    log_dir = os.path.join(
        "logs",
        f"{args.domain_name}-{args.task_name}",
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )

    algo = SlacAlgorithm(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=args.action_repeat,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
    )
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=args.seed,
        num_steps=args.num_steps,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--domain_name", type=str, default="cheetah")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(args)
