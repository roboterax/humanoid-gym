# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
from torch.distributions import Normal
from collections import deque
import numpy as np
from humanoid.algo.teacher_student.nn_model.tcn import TorchDeque

class ActorCriticTeacher(nn.Module):
    is_recurrent = False
    is_teacher = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        num_latent_embeddings=72,
                        num_encoder_info=72,
                        encoder_hidden_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation = nn.ELU(),
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        mlp_input_dim_e = num_encoder_info
        mlp_input_dim_a = num_actor_obs + num_latent_embeddings
        mlp_input_dim_c = num_critic_obs

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # Policy
        self.actor_nn = actor_nn(mlp_input_dim_a, mlp_input_dim_e, actor_hidden_dims, encoder_hidden_dims)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_mlp = nn.Sequential(*critic_layers)

        print(f"Critic MLP: {self.critic_mlp}")

        

    def actor(self, obs, privileged_obs):

        mean = self.actor_nn(obs, privileged_obs)
        return mean
    
    def critic(self, obs, privileged_obs):
        critic_obs = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic_mlp(critic_obs)

    def update_distribution(self, observations, privileged_observations):
        
        mean = self.actor(observations, privileged_observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, privileged_observations, **kwargs):
        self.update_distribution(observations, privileged_observations)
        return self.distribution.sample()

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, privileged_observations):
        
        actions_mean = self.actor(observations, privileged_observations)
        return actions_mean

    def evaluate(self, observations, privileged_observations, **kwargs):
        # print(critic_observations.shape)
        value = self.critic(observations, privileged_observations)
        return value

class actor_nn(nn.Module):
    def __init__(self, mlp_input_dim_a, mlp_input_dim_e, actor_hidden_dims, encoder_hidden_dims, num_latent_embeddings=72, activation=nn.ELU(), num_actions=12):
        super().__init__()
        # Policy
        # Actor MLP
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_mlp = nn.Sequential(*actor_layers)

        # MLP Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_e, encoder_hidden_dims[0]))
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], num_latent_embeddings))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        self.latent_embeddings = None

    def forward(self, obs, privileged_obs):
        self.latent_embeddings = self.encoder(privileged_obs)
        actor_input = torch.cat((obs, self.latent_embeddings), dim=-1)
        mean = self.actor_mlp(actor_input)
        return mean

class Student(nn.Module):
    def __init__(self,
                 tcn, 
                 mlp, 
                 perceptive_field_len, 
                 num_envs, 
                 num_actions=12,
                 init_noise_std=1.0,
                 rl_device='cuda:0'):
        super(Student, self).__init__()
        self.tcn = tcn
        self.mlp = mlp 
        self.perceptive_field_len = perceptive_field_len
        self.perceptive_field = TorchDeque(max_len=perceptive_field_len)
        self.pf_tensor = None
        self.device = rl_device
        self.num_envs = num_envs
        self.init_obs_buf()
        self.linear = nn.Linear(20, 1)
        self.tanh = nn.Tanh()
        self.latent_embeddings = torch.empty((0, ))

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))


    def forward(self, obs):
        self.perceptive_field.enqueue(obs[:, :36])
        # for i in self.perceptive_field:
        #     print(i.shape)
        pf_input = self.perceptive_field.queue.view(-1, 36, self.num_envs).to(self.device)
        latent_embeddings = self.tcn(pf_input)
        latent_embeddings = self.linear(latent_embeddings.view(self.num_envs, 72, 20)).view(self.num_envs, -1)
        latent_embeddings = self.tanh(latent_embeddings)
        self.latent_embeddings = latent_embeddings
        action = self.mlp(torch.concat((obs, latent_embeddings), dim=-1))
        return action
    
    def init_obs_buf(self):
        self.perceptive_field.enqueue([torch.zeros((self.num_envs, 36)).to(self.device) for _ in range(self.perceptive_field_len)])

    def update_distribution(self, observations):
        
        mean = self.forward(observations)
        
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def act_inference(self, observations):
        
        actions_mean = self.forward(observations)
        return actions_mean

