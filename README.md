# Stochastic Latent Actor-Critic in PyTorch
A PyTorch implementation of Stochastic Latent Actor-Critic[[1]](#references) for [DeepMind Control Suite](https://github.com/deepmind/dm_control). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

**UPDATE**
- 2021.10.6
    - Refactor codes to be compatible with original impl.
- 2020.10.26
    - Refactor codes and speed up training.
- 2020.8.28
    - Bump torch up to 1.6.0.

## Requirements
You can install Python liblaries using `pip install -r requirements.txt`. Note that you need a MuJoCo license. Please follow the instruction in [mujoco-py](https://github.com/openai/mujoco-py) for help.

If you're using other than CUDA 10.2, please install PyTorch following [instructions](https://pytorch.org/get-started/locally/) here.


## Examples
You can train SLAC algorithm as shown in the following example. Hyperparameters except action repeat are constant across all tasks. Please refer to Appendix B of the paper for more details.

```
python train.py --domain_name cheetah --task_name run --action_repeat 4 --seed 0 --cuda
```

Results (averaged over 2 seeds) on `cheetah-run` and `walker-walk` are as follows. Note that the horizontal axis represents environment steps, which equals to agent's steps multiplied by action repeat.

<img src="https://user-images.githubusercontent.com/37267851/136091614-bf36f6e2-991a-45d1-8b8e-8f22718dbbe3.png" width=410>  <img src="https://user-images.githubusercontent.com/37267851/136091624-1bfcf519-3697-4b1e-aad0-4b5211fc64e2.png" width=410>

Visualization of image sequence corresponding to Figure 9 in the paper is as follows. First row is ground truth, second row is generated image from posterior sample (from the latent model), third row is generated image from prior sample only conditioned on the initial frame and last row is generated image from prior sample. Please refer to the paper for details.

<img src="https://user-images.githubusercontent.com/37267851/69476615-6802a400-0e1f-11ea-919d-b7958413efab.png" title="sequence" width=750>

## References
[[1]](https://arxiv.org/abs/1907.00953) Lee, Alex X., et al. "Stochastic latent actor-critic: Deep reinforcement learning with a latent variable model." arXiv preprint arXiv:1907.00953 (2019).
