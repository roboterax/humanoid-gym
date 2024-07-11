from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }
    
    class env(LeggedRobotCfg.env):
        num_envs = 1500
        frame_stack = 40
        c_frame_stack = 3
        num_single_obs = 66
        num_actions = 10
        num_observations = num_single_obs # int(frame_stack * num_single_obs) for MLP 
        num_teaching_observations = int(frame_stack * (num_single_obs-1))
        single_num_privileged_obs = 65
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        use_ref_actions = False
        episode_length_s = 60  # episode length in seconds
      

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        curriculum = True

        class ranges:
            lin_vel_x = [-1.0, 2.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1,0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand:
        randomize_friction = False
        friction_range = [0.1, 2.0]
        randomize_base_mass = False
        added_mass_range = [-5., 5.]
        push_robots = False
        push_interval_s = 4
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset( LeggedRobotCfg.asset ):
        file = '/home/ziluoding/humanoid-gym/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards:
        soft_dof_pos_limit = 0.9
        base_height_target = 0.98
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06       # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # forces above this value are penalized

        class scales:
            # reference motion tracking
            # joint_pos = 1.6
            # feet_clearance = 1.
            # feet_contact_number = 1.2
            # # gait
            # feet_air_time = 1.
            # foot_slip = -0.05
            # feet_distance = 1
            # knee_distance = 1
            # # contact
            # feet_contact_forces = -0.01
            # # vel tracking
            # tracking_lin_vel = 1.2 
            # tracking_ang_vel = 1.1
            # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # low_speed = 0.2
            # track_vel_hard = 0.5
            # # base pos
            # default_joint_pos = 0.5
            # orientation = 1.
            # base_height = 0.2
            # base_acc = 0.2
            # # energy
            # action_smoothness = -0.002
            # torques = -1e-5
            # dof_vel = -5e-4
            # dof_acc = -1e-7
            # collision = -1.

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -1.0
            orientation = -1.0
            base_height = -100.0
            dof_acc = -3.5e-8
            feet_air_time = 1.0
            collision = 0.0
            action_rate = -0.01
            torques = 0.0
            dof_pos_limits = -10.0

        class stand_scales:
            # reference motion tracking
            joint_pos = 1.6 * 0
            feet_clearance = 1. * 0
            feet_contact_number = 1.2
            # gait
            feet_air_time = -1*0.
            foot_slip = 0
            feet_distance = 2
            knee_distance = 2
            # contact
            feet_contact_forces = -0.01 * 0
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            #low_speed = 0.2
            track_vel_hard = 0.5 * 1
            # base pos
            default_joint_pos = 0.5 * 0
            orientation = 1. * 1 * 0
            base_height = 0.2 * 2
            base_acc = 0.2 * 1
            # energy
            action_smoothness = -0.02
            torques = -1e-5
            dof_vel = -5e-3
            dof_acc = -1e-4
            collision = -1.

class H1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        policy_type = 'moving'
        teaching_model_path = '/home/ps/humanoid-gym/logs/h1/MLP_best/model_15000.pt'
        # For LSTM only
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'h1'
        policy_class_name = 'ActorCritic'

  
