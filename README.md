# Stochastic Latent Actor-Critic in PyTorch
A PyTorch implementation of Stochastic Latent Actor-Critic[[1]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

## Requirements
You can install liblaries using `pip install -r requirements.txt` except [mujoco_py](https://github.com/openai/mujoco-py) and [dm_control](https://github.com/deepmind/dm_control).

Note that you need a licence to install mujoco_py. For installation, please follow instructions [here](https://github.com/deepmind/dm_control).

## Examples
You can train SLAC agent on the task from DeepMind Control Suite like this example [here](https://github.com/ku2482/slac.pytorch/blob/master/code/main.py). Hyperparameters except `action_repeat` are constant across all tasks. Please refer to Appendix B of the paper for details.

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

## Results
Result of above example on Cheetah run and Walker walk tasks are below. Note that the horizontal axis represents environment steps, which equals agent's steps multiplied by `action_repeat`.

<img src="https://user-images.githubusercontent.com/37267851/69509282-c3e23f80-0f7b-11ea-8862-299872a5b89b.png" width=420><img src="https://user-images.githubusercontent.com/37267851/69509339-f724ce80-0f7b-11ea-8df8-eb5d8aaa57fa.png" width=420>

Visualizations of image sequence corresponding to Figure 9 of the paper are below. First row is ground truth, second row is generated image from posterior sample (of the latent model), third row is generated image from prior sample only conditioned on the initial frame and last row is generated image from prior sample. Please refer to the paper for details.

<img src="https://user-images.githubusercontent.com/37267851/69476615-6802a400-0e1f-11ea-919d-b7958413efab.png" title="sequence" width=750>

## References
[[1]](https://arxiv.org/abs/1907.00953) Lee, Alex X., et al. "Stochastic latent actor-critic: Deep reinforcement learning with a latent variable model." arXiv preprint arXiv:1907.00953 (2019).
