# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

import numpy as np


class MrobotCfgTS(LeggedRobotCfg):
    """
    Configuration class for the L1 humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        # frame_stack = 15
        # c_frame_stack = 3
        frame_stack = 1
        c_frame_stack = 1
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        # single_num_privileged_obs = 73
        single_num_privileged_obs = 121
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions


    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.95

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Mrobot/urdf/L1.urdf'

        name = "L1"
        foot_name = "6_link"
        ankle_name = "5_link"
        knee_name = "4_link"

        terminate_after_contacts_on = [
            'base_link',
            'leg_l1_link',
            'leg_l2_link',
            'leg_l3_link',
            'leg_l4_link',
            'leg_l5_link',

            'leg_r1_link',
            'leg_r2_link',
            'leg_r3_link',
            'leg_r4_link',
            'leg_r5_link',
        ]
        penalize_contacts_on = [
            'base_link',
            'leg_l1_link',
            'leg_l2_link',
            'leg_l3_link',
            'leg_l4_link',
            'leg_l5_link',
            
            'leg_r1_link',
            'leg_r2_link',
            'leg_r3_link',
            'leg_r4_link',
            'leg_r5_link',
        ]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = True
        # rough terrain only:
        measure_heights = True
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        # add_noise = False
        noise_level = 0.6    # scales other values

        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.2
            lin_vel = 0.1
            quat = 0.1
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0., 0., 1.1]

        default_joint_angles = {
            'leg_l1_joint': 0.077,
            'leg_l2_joint': -0.024,
            'leg_l3_joint': -0.06,
            'leg_l4_joint': 0.15,
            'leg_l5_joint': -0.11,
            'leg_l6_joint': 0.024,

            'leg_r1_joint': -0.077,
            'leg_r2_joint': 0.024,
            'leg_r3_joint': 0.06,
            'leg_r4_joint': 0.15,
            'leg_r5_joint': -0.11,
            'leg_r6_joint': -0.024,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'leg_l1_joint': 280, #105
            'leg_l2_joint': 500, #105
            'leg_l3_joint': 200.0, #75
            'leg_l4_joint': 250,
            'leg_l5_joint': 40,
            'leg_l6_joint': 20,

            'leg_r1_joint': 280,
            'leg_r2_joint': 500,
            'leg_r3_joint': 200,
            'leg_r4_joint': 250,
            'leg_r5_joint': 40,
            'leg_r6_joint': 20,
          }
        damping = {
            'leg_l1_joint': 28,
            'leg_l2_joint': 32,
            'leg_l3_joint': 20,
            'leg_l4_joint': 25,
            'leg_l5_joint': 4,
            'leg_l6_joint': 2,

            'leg_r1_joint': 28,
            'leg_r2_joint': 32,
            'leg_r3_joint': 20,
            'leg_r4_joint': 25,
            'leg_r5_joint': 4,
            'leg_r6_joint': 2,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]

        randomize_base_mass = True
        added_mass_range = [-5., 5.]

        randomize_com_displacement = True
        com_displacement_range = [-0.06, 0.06]
        com_offset_x = -0.15

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]

        randomize_friction = True
        friction_range = [0.1, 2.0]

        randomize_restitution = True
        restitution_range = [0., 0.5]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035]

        randomize_joint_friction = False
        joint_friction_range = [[0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0],

                                [0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0]]

        randomize_joint_armature = True
        joint_armature_range = [[0., 0.01],
                                [0., 0.01],
                                [0., 0.01],
                                [0., 0.01],
                                [0., 0.01],
                                [0., 0.01],

                                [0., 0.01],
                                [0., 0.01],
                                [0., 0.01],
                                [0., 0.01],
                                [0., 0.01],
                                [0., 0.01]]

        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [0., 0.7]  # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.89
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.
       
    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class MrobotCfgPPOTS(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunnerTeacher'   # DWLOnPolicyRunner


    class policy:
        # init_noise_std = 1.0
        fixed_std = True
        init_noise_std = 0.27
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        # encoder_hidden_dims = [768, 256, 128]


    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        schedule = 'adaptive' # could be adaptive, fixed
        desired_kl = 0.01
        max_grad_norm = 1.
        num_learning_epochs = 4
        gamma = 0.995
        lam = 0.95
        num_mini_batches = 5
        use_clipped_value_loss = True
        value_loss_coef = 1.0
        clip_param = 0.1

    class runner:
        policy_class_name = 'ActorCriticTeacher'
        algorithm_class_name = 'PPO_teacher'
        num_steps_per_env = 60  # per iteration
        max_iterations = 30000  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'L1_ts'
        run_name = ''
        # Load and resume
        resume = True
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt