# Stochastic Latent Actor-Critic in PyTorch
A PyTorch implementation of Stochastic Latent Actor-Critic[[1]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

## Requirements
You can install liblaries using `pip install -r requirements.txt` except [mujoco_py](https://github.com/openai/mujoco-py) and [dm_control](https://github.com/deepmind/dm_control).

Note that you need a licence to install mujoco_py. For installation, please follow instructions [here](https://github.com/deepmind/dm_control).

## Examples
You can train SLAC agent on the task from DeepMind Control Suite like this example [here](https://github.com/ku2482/slac.pytorch/blob/master/code/main.py).

```
python code/main.py \
--env_type dm_control \
--domain_name cheetah \
--task_name run \
--action_repeat 4 \
--seed 0 \
--cuda (optional)
```

I haven't tested on the OpenAI Gym(MuJoCo) benchmark tasks due to OpenGL issues.

## References
[[1]](https://arxiv.org/abs/1907.00953) Lee, Alex X., et al. "Stochastic latent actor-critic: Deep reinforcement learning with a latent variable model." arXiv preprint arXiv:1907.00953 (2019).
