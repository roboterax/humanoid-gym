# <a href="https://sites.google.com/view/humanoid-gym/">Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer</a>

<a href="https://sites.google.com/view/humanoid-gym/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2404.05695"><strong>arXiv</strong></a>
  |
  <a href="https://twitter.com/roboterax/status/1765038672641175662"><strong>Twitter</strong></a>

  <a href="https://github.com/zlw21gxy">Xinyang Gu*</a>, 
  <a href="https://wangyenjen.github.io/">Yen-Jen Wang*</a>,
  <a href="http://people.iiis.tsinghua.edu.cn/~jychen/">Jianyu Chen†</a>

  *: Equal contribution. Project Co-lead., †: Corresponding Author.

![Demo](./images/demo.gif)

Humanoid-Gym is an easy-to-use reinforcement learning (RL) framework based on Nvidia Isaac Gym, designed to train locomotion skills for humanoid robots, emphasizing zero-shot transfer from simulation to the real-world environment. Humanoid-Gym also integrates a sim-to-sim framework from Isaac Gym to Mujoco that allows users to verify the trained policies in different physical simulations to ensure the robustness and generalization of the policies.

This codebase is verified by RobotEra's XBot-S (1.2-meter tall humanoid robot) and XBot-L (1.65-meter tall humanoid robot) in a real-world environment with zero-shot sim-to-real transfer.

## Features

### 1. Humanoid Robot Training
This repository offers comprehensive guidance and scripts for the training of humanoid robots. Humanoid-Gym features specialized rewards for humanoid robots, simplifying the difficulty of sim-to-real transfer. In this repository, we use RobotEra's XBot-L as a primary example. It can also be used for other robots with minimal adjustments. Our resources cover setup, configuration, and execution. Our goal is to fully prepare the robot for real-world locomotion by providing in-depth training and optimization.


- **Comprehensive Training Guidelines**: We offer thorough walkthroughs for each stage of the training process.
- **Step-by-Step Configuration Instructions**: Our guidance is clear and succinct, ensuring an efficient setup process.
- **Execution Scripts for Easy Deployment**: Utilize our pre-prepared scripts to streamline the training workflow.

### 2. Sim2Sim Support
We also share our sim2sim pipeline, which allows you to transfer trained policies to highly accurate and carefully designed simulated environments. Once you acquire the robot, you can confidently deploy the RL-trained policies in real-world settings.

Our simulator settings, particularly with Mujoco, are finely tuned to closely mimic real-world scenarios. This careful calibration ensures that the performances in both simulated and real-world environments are closely aligned. This improvement makes our simulations more trustworthy and enhances our confidence in their applicability to real-world scenarios.


### 3. Denoising World Model Learning
#### Robotics: Science and Systems (RSS), 2024 (Best Paper Award Finalist)
<a href="https://enriquecoronadozu.github.io/rssproceedings2024/rss20/p058.pdf"><strong>Paper</strong></a>
|
<a href="https://x.com/wangyenjen/status/1792741940087394540"><strong>Twitter</strong></a>

<a href="https://github.com/zlw21gxy">Xinyang Gu*</a>, 
<a href="https://wangyenjen.github.io/">Yen-Jen Wang*</a>,
Xiang Zhu*, Chengming Shi*, Yanjiang Guo, Yichen Liu,
<a href="http://people.iiis.tsinghua.edu.cn/~jychen/">Jianyu Chen†</a>

*: Equal contribution. Project Co-lead., †: Corresponding Author.

Denoising World Model Learning(DWL) presents an advanced sim-to-real framework that integrates state estimation and system identification. This dual-method approach ensures the robot's learning and adaptation are both practical and effective in real-world contexts.

- **Enhanced Sim-to-real Adaptability**: Techniques to optimize the robot's transition from simulated to real environments.
- **Improved State Estimation Capabilities**: Advanced tools for precise and reliable state analysis.

### Perceptive Locomotion Learning for Humanoid Robots (Coming Soon!)
<a href="https://x.com/roboterax/status/1798694054374564010"><strong>Twitter</strong></a>

### Dexterous Hand Manipulation (Coming Soon!)
<a href="https://x.com/roboterax/status/1791349763448938924"><strong>Twitter</strong></a>

## Installation

1. Generate a new Python virtual environment with Python 3.8 using `conda create -n myenv python=3.8`.
2. For the best performance, we recommend using NVIDIA driver version 525 `sudo apt install nvidia-driver-525`. The minimal driver version supported is 515. If you're unable to install version 525, ensure that your system has at least version 515 to maintain basic functionality.
3. Install PyTorch 1.13 with Cuda-11.7:
   - `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
4. Install numpy-1.23 with `conda install numpy=1.23`.
5. Install Isaac Gym:
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym.
   - `cd isaacgym/python && pip install -e .`
   - Run an example with `cd examples && python 1080_balls_of_solitude.py`.
   - Consult `isaacgym/docs/index.html` for troubleshooting.
6. Install humanoid-gym:
   - Clone this repository.
   - `cd humanoid-gym && pip install -e .`



## Usage Guide

#### Examples

```bash
# Under the directory humanoid-gym/humanoid
# Launching PPO Policy Training for 'v1' Across 4096 Environments
# This command initiates the PPO algorithm-based training for the humanoid task.
python scripts/train.py --task=humanoid_ppo --run_name v1 --headless --num_envs 4096

# Evaluating the Trained PPO Policy 'v1'
# This command loads the 'v1' policy for performance assessment in its environment. 
# Additionally, it automatically exports a JIT model, suitable for deployment purposes.
python scripts/play.py --task=humanoid_ppo --run_name v1

# Implementing Simulation-to-Simulation Model Transformation
# This command facilitates a sim-to-sim transformation using exported 'v1' policy.
# You have to run play.py first to get the JIT model and use it with sim2sim.py
python scripts/sim2sim.py --load_model /path/to/logs/XBot_ppo/exported/policies/policy_1.pt

# Run our trained policy
python scripts/sim2sim.py --load_model /path/to/logs/XBot_ppo/exported/policies/policy_example.pt

```

#### 1. Default Tasks


- **humanoid_ppo**
   - Purpose: Baseline, PPO policy, Multi-frame low-level control
   - Observation Space: Variable $(47 \times H)$ dimensions, where $H$ is the number of frames
   - $[O_{t-H} ... O_t]$
   - Privileged Information: $73$ dimensions

- **humanoid_dwl (coming soon)**

#### 2. PPO Policy
- **Training Command**: For training the PPO policy, execute:
  ```
  python humanoid/scripts/train.py --task=humanoid_ppo --load_run log_file_path --name run_name
  ```
- **Running a Trained Policy**: To deploy a trained PPO policy, use:
  ```
  python humanoid/scripts/play.py --task=humanoid_ppo --load_run log_file_path --name run_name
  ```
- By default, the latest model of the last run from the experiment folder is loaded. However, other run iterations/models can be selected by adjusting `load_run` and `checkpoint` in the training config.

#### 3. Sim-to-sim
- **Please note: Before initiating the sim-to-sim process, ensure that you run `play.py` to export a JIT policy.**
- **Mujoco-based Sim2Sim Deployment**: Utilize Mujoco for executing simulation-to-simulation (sim2sim) deployments with the command below:
  ```
  python scripts/sim2sim.py --load_model /path/to/export/model.pt
  ```


#### 4. Parameters
- **CPU and GPU Usage**: To run simulations on the CPU, set both `--sim_device=cpu` and `--rl_device=cpu`. For GPU operations, specify `--sim_device=cuda:{0,1,2...}` and `--rl_device={0,1,2...}` accordingly. Please note that `CUDA_VISIBLE_DEVICES` is not applicable, and it's essential to match the `--sim_device` and `--rl_device` settings.
- **Headless Operation**: Include `--headless` for operations without rendering.
- **Rendering Control**: Press 'v' to toggle rendering during training.
- **Policy Location**: Trained policies are saved in `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`.

#### 5. Command-Line Arguments
For RL training, please refer to `humanoid/utils/helpers.py#L161`.
For the sim-to-sim process, please refer to `humanoid/scripts/sim2sim.py#L169`.

## Code Structure

1. Every environment hinges on an `env` file (`legged_robot.py`) and a `configuration` file (`legged_robot_config.py`). The latter houses two classes: `LeggedRobotCfg` (encompassing all environmental parameters) and `LeggedRobotCfgPPO` (denoting all training parameters).
2. Both `env` and `config` classes use inheritance.
3. Non-zero reward scales specified in `cfg` contribute a function of the corresponding name to the sum-total reward.
4. Tasks must be registered with `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. Registration may occur within `envs/__init__.py`, or outside of this repository.


## Add a new environment 

The base environment `legged_robot` constructs a rough terrain locomotion task. The corresponding configuration does not specify a robot asset (URDF/ MJCF) and no reward scales.

1. If you need to add a new environment, create a new folder in the `envs/` directory with a configuration file named `<your_env>_config.py`. The new configuration should inherit from existing environment configurations.
2. If proposing a new robot:
    - Insert the corresponding assets in the `resources/` folder.
    - In the `cfg` file, set the path to the asset, define body names, default_joint_positions, and PD gains. Specify the desired `train_cfg` and the environment's name (python class).
    - In the `train_cfg`, set the `experiment_name` and `run_name`.
3. If needed, create your environment in `<your_env>.py`. Inherit from existing environments, override desired functions and/or add your reward functions.
4. Register your environment in `humanoid/envs/__init__.py`.
5. Modify or tune other parameters in your `cfg` or `cfg_train` as per requirements. To remove the reward, set its scale to zero. Avoid modifying the parameters of other environments!
6. If you want a new robot/environment to perform sim2sim, you may need to modify `humanoid/scripts/sim2sim.py`: 
    - Check the joint mapping of the robot between MJCF and URDF.
    - Change the initial joint position of the robot according to your trained policy.

## Troubleshooting

Observe the following cases:

```bash
# error
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

# solution
# set the correct path
export LD_LIBRARY_PATH="~/miniconda3/envs/your_env/lib:$LD_LIBRARY_PATH" 

# OR
sudo apt install libpython3.8

# error
AttributeError: module 'distutils' has no attribute 'version'

# solution
# install pytorch 1.12.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# error, results from libstdc++ version distributed with conda differing from the one used on your system to build Isaac Gym
ImportError: /home/roboterax/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20` not found (required by /home/roboterax/carbgym/python/isaacgym/_bindings/linux64/gym_36.so)

# solution
mkdir ${YOUR_CONDA_ENV}/lib/_unused
mv ${YOUR_CONDA_ENV}/lib/libstdc++* ${YOUR_CONDA_ENV}/lib/_unused
```

## Citation

Please cite the following if you use this code or parts of it:
```
@article{gu2024humanoid,
  title={Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer},
  author={Gu, Xinyang and Wang, Yen-Jen and Chen, Jianyu},
  journal={arXiv preprint arXiv:2404.05695},
  year={2024}
}

@inproceedings{gu2024advancing,
  title={Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning},
  author={Gu, Xinyang and Wang, Yen-Jen and Zhu, Xiang and Shi, Chengming and Guo, Yanjiang and Liu, Yichen and Chen, Jianyu},
  booktitle={Robotics: Science and Systems},
  year={2024},
  url={https://enriquecoronadozu.github.io/rssproceedings2024/rss20/p058.pdf}
}
```

## Acknowledgment

The implementation of Humanoid-Gym relies on resources from [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) projects, created by the Robotic Systems Lab. We specifically utilize the `LeggedRobot` implementation from their research to enhance our codebase.

## Any Questions?

If you have any more questions, please contact [support@robotera.com](mailto:support@robotera.com) or create an issue in this repository.
