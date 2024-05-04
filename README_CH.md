Humanoid-Gym：采用零样本 Sim2Real 迁移的人形机器人强化学习  
**[[论文]](https://arxiv.org/abs/2404.05695)**
**[[项目页面]](https://sites.google.com/view/humanoid-gym/)**
**[[English]](./README.md)**

![演示](./images/demo.gif)

欢迎来到我们的Humanoid-Gym！

Humanoid-Gym 是一个基于 Nvidia Isaac Gym 的易于使用的强化学习 (RL) 框架，旨在训练人形机器人的运动技能，强调从模拟到现实环境的零样本迁移。 Humanoid-Gym还集成了从Isaac Gym到Mujoco的sim-to-sim框架，允许用户在不同的物理模拟中验证训练好的策略，以确保策略的鲁棒性和泛化性。

该代码库由 RobotEra 的 XBot-S（1.2 米高的人形机器人）和 XBot-L（1.65 米高的人形机器人）在现实环境中进行了验证，具有零样本模拟到真实的传输。

## 特征
### 1. 仿人机器人训练
该存储库为人形机器人的训练提供全面的指导和脚本。 Humanoid-Gym 为人形机器人提供专门的奖励，简化了模拟到现实的转换难度。在此存储库中，我们使用 RobotEra 的 XBot-L 作为主要示例。只需进行最少的调整，它也可用于其他机器人。我们的资源涵盖设置、配置和执行。我们的目标是通过提供深入的训练和优化，让机器人为现实世界的运动做好充分准备。

- **全面的训练指南**：我们为训练过程的每个阶段提供全面的演练。
- **分步配置说明**：我们的指导清晰简洁，确保设置过程高效。
- **用于轻松部署的执行脚本**：利用我们预先准备的脚本来简化训练工作流程。
### 2.Sim2Sim支持
我们还共享我们的 sim2sim 管道，它允许您将训练有素的策略转移到高度准确且精心设计的模拟环境中。获得机器人后，您就可以自信地在现实环境中部署强化学习训练的策略。

我们的模拟器设置（尤其是 Mujoco）经过精心调整，可以紧密模仿现实世界的场景。这种仔细的校准可确保模拟环境和现实环境中的性能紧密结合。这一改进使我们的模拟更加值得信赖，并增强了我们对其适用于现实场景的信心。

### 3.去噪世界模型学习（即将推出！）
去噪世界模型学习（DWL）提出了一种先进的模拟到真实的框架，集成了状态估计和系统识别。这种双重方法确保机器人的学习和适应在现实环境中既实用又有效。

- **增强的模拟到真实的适应性**：优化机器人从模拟环境到真实环境的过渡的技术。
- **改进的状态估计功能**：用于精确可靠的状态分析的高级工具。
### 灵巧的手部操作（即将推出！）
## 安装
1. 使用 Python 3.8 生成新的 Python 虚拟环境`conda create -n myenv python=3.8`。
2. 为了获得最佳性能，我们建议使用 NVIDIA 驱动程序版本 525 `sudo apt install nvidia-driver-525`。支持的最低驱动程序版本为 515。如果您无法安装版本 525，请确保您的系统至少具有版本 515 以维持基本功能。
3. 安装 PyTorch 1.13 和 Cuda-11.7：
- `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
4. 安装 numpy-1.23 和conda install numpy=1.23.
安装Isaac Gym Preview 4：
从https://developer.nvidia.com/isaac-gym下载并安装 Isaac Gym Preview 4 。
- `cd isaacgym/python && pip install -e .`
- 运行一个示例`cd examples && python 1080_balls_of_solitude.py`。
- 请咨询`isaacgym/docs/index.html`故障排除。
6. 安装humanoid-gym：
- 克隆这个存储库。
- `cd humanoid_gym && pip install -e .`

## 使用指南
#### 示例
```bash
# 在4096环境中启动"v1"的PPO策略训练
# 该命令启动基于 PPO 算法的人形任务训练
python scripts/train.py --task=humanoid_ppo --run_name v1 --headless --num_envs 4096

# 评估经过训练的 PPO 政策“v1”
# 此命令加载“v1”策略以在其环境中进行性能评估。
# 此外，它会自动导出 JIT 模型，适合部署目的。
python scripts/play.py --task=humanoid_ppo --run_name v1

# Implementing Simulation-to-Simulation Model Transformation
# This command facilitates a sim-to-sim transformation using exported 'v1' policy.
python scripts/sim2sim.py --load_model /path/to/logs/XBot_ppo/exported/policies/policy_1.pt

# 运行已训练好的policy
python scripts/sim2sim.py --load_model /path/to/logs/XBot_ppo/exported/policies/policy_example.pt
```
#### 1. 默认任务


- **humanoid_ppo**
   - 目的: 基准线, PPO策略, 多帧低级控制
   - 观察空间: 可变 $(47 \times H)$ 维度, 其中 $H$ 是帧数
   - $[O_{t-H} ... O_t]$
   - 特权信息: $73$ 维

- **humanoid_dwl (即将推出)**

#### 2. PPO 策略
- **训练命令**: 要训练PPO策略,执行:
  ```
  python humanoid/scripts/train.py --task=humanoid_ppo --load_run log_file_path --name run_name
  ```


- **运行训练好的策略**: 要部署训练好的PPO策略,使用:
  ```
  python humanoid/scripts/play.py --task=humanoid_ppo --load_run log_file_path --name run_name
  ```

- 默认情况下,会加载实验文件夹中最后一次运行的最新模型。但是,可以通过调整训练配置中的`load_run`和`checkpoint`来选择其他运行迭代/模型。

#### 3. Sim-to-sim

- **基于Mujoco的Sim2Sim部署**: 使用以下命令利用Mujoco执行从模拟到模拟(sim2sim)的部署:
  ```
  python scripts/sim2sim.py --load_model /path/to/export/model.pt
  ```



#### 4. 参数
- **CPU和GPU使用**: 要在CPU上运行模拟,请同时设置`--sim_device=cpu`和`--rl_device=cpu`。对于GPU操作,请相应地指定`--sim_device=cuda:{0,1,2...}`和`--rl_device={0,1,2...}`。请注意,`CUDA_VISIBLE_DEVICES`不适用,并且必须匹配`--sim_device`和`--rl_device`设置。
- **无头操作**: 对于无渲染操作,请包括`--headless`。
- **渲染控制**: 在训练过程中按"v"切换渲染。
- **策略位置**: 训练好的策略保存在`humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`中。

#### 5. 命令行参数
对于RL训练,请参考`humanoid/utils/helpers.py#L161`。
对于sim-to-sim过程,请参考`humanoid/scripts/sim2sim.py#L169`。

## 代码结构

1. 每个环境都依赖于一个`env`文件(`legged_robot.py`)和一个`configuration`文件(`legged_robot_config.py`)。后者包含两个类:`LeggedRobotCfg`(包含所有环境参数)和`LeggedRobotCfgPPO`(表示所有训练参数)。
2. `env`和`config`类都使用继承。
3. 在`cfg`中指定的非零奖励尺度为总奖励的求和贡献了一个具有相应名称的函数。
4. 任务必须使用`task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`进行注册。注册可能发生在`envs/__init__.py`中,或者在此存储库之外。


## 添加新环境

基础环境`legged_robot`构建了一个崎岖地形运动任务。相应的配置没有指定机器人资产(URDF/MJCF),也没有奖励尺度。

1. 如果需要添加新环境,请在`envs/`目录中创建一个新文件夹,其中包含一个名为`<your_env>_config.py`的配置文件。新配置应该继承现有环境配置。
2. 如果提出一个新机器人:
  - 将相应的资产插入`resources/`文件夹。
  - 在`cfg`文件中,设置资产的路径,定义主体名称、default_joint_positions和PD增益。指定所需的`train_cfg`和环境的名称(python类)。
  - 在`train_cfg`中,设置`experiment_name`和`run_name`。
3. 如果需要,在`<your_env>.py`中创建你的环境。继承现有环境,覆盖所需的函数和/或添加你的奖励函数。
4. 在`humanoid/envs/__init__.py`中注册你的环境。
5. 根据需要修改或调整`cfg`或`cfg_train`中的其他参数。要移除奖励,请将其尺度设置为零。避免修改其他环境的参数!
6. 如果你想要一个新的机器人/环境执行sim2sim,你可能需要修改`humanoid/scripts/sim2sim.py`: 
  - 检查MJCF和URDF之间机器人的关节映射。
  - 根据你训练的策略更改机器人的初始关节位置。

## 故障排除

观察以下情况:

```bash
# 错误
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

# 解决方案
# 设置正确的路径
export LD_LIBRARY_PATH="~/miniconda3/envs/your_env/lib:$LD_LIBRARY_PATH" 

# 或者
sudo apt install libpython3.8

# 错误
AttributeError: module 'distutils' has no attribute 'version'

# 解决方案
# 安装pytorch 1.12.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 错误,由于conda分发的libstdc++版本与系统用于构建Isaac Gym的版本不同而导致
ImportError: /home/roboterax/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20` not found (required by /home/roboterax/carbgym/python/isaacgym/_bindings/linux64/gym_36.so)

# 解决方案
mkdir ${YOUR_CONDA_ENV}/lib/_unused
mv ${YOUR_CONDA_ENV}/lib/libstdc++* ${YOUR_CONDA_ENV}/lib/_unused 
```
## 引用
如果你使用此代码或其部分,请引用以下内容:
```
@article{gu2024humanoid,
  title={Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer},
  author={Gu, Xinyang and Wang, Yen-Jen and Chen, Jianyu},
  journal={arXiv preprint arXiv:2404.05695},
  year={2024}
}
```

## 致谢  
Humanoid-Gym的实现依赖于Robotic Systems Lab创建的[legged_gym](https://github.com/leggedrobotics/legged_gym) 和[rsl_rl](https://github.com/leggedrobotics/rsl_rl) 项目的资源。我们特别利用他们研究中的`LeggedRobot` 实现来增强我们的代码库。

## 相关问题  
如遇到任何问题，请联系[support@robotera.com](mailto:support@robotera.com)
