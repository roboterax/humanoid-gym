import torch 
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import ConcatDataset
from torch.nn import MSELoss
from torch.autograd import Variable

import numpy as np

from humanoid.algo.vec_env import VecEnv
from humanoid.algo.teacher_student.nn_model.tcn import TemporalConvNet as TCN
from humanoid.algo.teacher_student.actor_critic_teacher import Student

from collections import deque 
import time
import os 


class DAgger:
    def __init__(self, teacher, env: VecEnv, rl_device):
        self.beta = 1
        self.seq_len = 20            # seq_len
        self.T_step = 50
        
        self.tcn = TCN(num_inputs=36, num_channels=[72 for _ in range(6)])
        self.teacher = teacher      # contains encoder and mlp
        self.student = Student(self.tcn, self.teacher.actor_mlp, self.seq_len, env.num_envs).to(rl_device)    # contains TCN and teacher_mlp
        self.obs_seq = deque(maxlen=self.seq_len)
        self.env = env
        self.device = rl_device
        self.optimizer = optim.Adam(self.student.parameters(), lr=5e-4, weight_decay=0.995)
        self.num_learning_iterations = 5
        self.decay_delta = 1/self.num_learning_iterations
        self.loss1 = MSELoss()
        self.loss2 = MSELoss()
        self.student_train_flag = 0

    # start the train iteration
    def train_iter(self):
        for i in range(self.num_learning_iterations):
            obs = self.env.get_observations().clone()
            privileged_obs = self.env.get_privileged_observations().clone()
            total_obs = obs.view(self.env.num_envs, obs.shape[1], 1)
            teacher_action = self.teacher.act(obs, privileged_obs)  
            total_teacher_action = teacher_action.view(self.env.num_envs, self.env.num_actions, 1)
            teacher_latent_embeddings = self.teacher.encoder(privileged_obs)
            total_teacher_latent_embeddings = teacher_latent_embeddings.view(self.env.num_envs, teacher_latent_embeddings.shape[1], 1)
            # Sample T-step trajectories using mixed policy
            for j in range(self.T_step):
                obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
                mix_action = self.mix_policy_act(obs, privileged_obs)
                # visited states by mix_policy
                obs, privileged_obs, _, _, _ = self.env.step(mix_action)
                # actions given by teacher
                teacher_action = self.teacher.act(obs, privileged_obs)  
                teacher_latent_embeddings = self.teacher.encoder(privileged_obs)
                # Append the i-th Dataset to Dataset
                total_obs = torch.concat((total_obs, obs.view(self.env.num_envs, obs.shape[1], 1)), dim=-1)
                total_teacher_action = torch.concat((total_teacher_action, teacher_action.view(self.env.num_envs, self.env.num_actions, 1)), dim=-1)
                total_teacher_latent_embeddings = torch.concat((total_teacher_latent_embeddings, teacher_latent_embeddings.view(self.env.num_envs, teacher_latent_embeddings.shape[1], 1)), dim=-1)
                    
            
            # Train the student
            self.student.train()
            
            for k in range(self.env.num_envs):
                obs = total_obs[:, :, k].view(self.env.num_envs, self.env.num_obs)
                a = total_teacher_action[:, :, k].view(self.env.num_envs, self.env.num_actions)
                l = total_teacher_latent_embeddings[:, :, k].view(self.env.num_envs, teacher_latent_embeddings.shape[1])
                obs, a = obs.to(self.device), a.to(self.device)   
                pred_a = self.student.act(obs)
                pred_l = self.student.latent_embeddings
                # a, pred_a, l, pred_l = Variable(a, requires_grad=True), Variable(pred_a, requires_grad=True), Variable(l, requires_grad=True), Variable(pred_l, requires_grad=True)
                loss1, loss2 = self.loss_func(a, pred_a, l, pred_l)
                loss1.backward()
                loss2.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.beta -= self.decay_delta

        self.student_train_flag = 1

    # mix policy
    def mix_policy_act(self, obs, privileged_obs):
        teacher_action = self.beta * self.teacher.act(obs, privileged_obs)
        student_action = (1-self.beta) * self.student(obs)
        return teacher_action + student_action

    def loss_func(self, output_actor_teacher, output_actor_student, output_encoder, output_tcn):
        return self.loss1(output_actor_teacher, output_actor_student),  self.loss2(output_encoder, output_tcn)


    
