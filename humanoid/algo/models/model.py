import numpy as np
import torch
import torch.nn as nn
from .models.transformer import GatedTransformerXL


class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])

class PPOTransformerModel(nn.Module):
    def __init__(self,config,state_size,action_size):
        super(PPOTransformerModel,self).__init__()
        """
        Overview:
            Init

        Arguments:
            - config: (`dict`): configuration.
            - state_size: (`int`): size of state.
            - action_size (`int`): size of action space
        Return:
        """
        self.fc = self._layer_init(nn.Linear(state_size,config["transformer"]['embed_dim']),std=np.sqrt(2))

        self.transformer = GatedTransformerXL(config,input_dim=config["transformer"]['embed_dim'])

        self.policy = nn.Sequential(
            nn.Tanh(),
            self._layer_init(nn.Linear(config["transformer"]['embed_dim'],config["transformer"]['hidden_size']),std=np.sqrt(2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(config["transformer"]['hidden_size'],action_size),std=0.01)
        )

        self.value = nn.Sequential(
            nn.Tanh(),
            self._layer_init(nn.Linear(config["transformer"]['embed_dim'],config["transformer"]['hidden_size']),std=np.sqrt(2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(config["transformer"]['hidden_size'],1),std=1)
        )

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        """
        Overview:
            Init Weight and Bias with Constraint

        Arguments:
            - layer: Layer.
            - std: (`float`): Standard deviation.
            - bias_const: (`float`): Bias

        Return:
        
        """
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self,state,memory,memory_mask,memory_indices):
        """
        Overview:
            Forward method.

        Arguments:
            - state: (torch.Tensor): state with shape (batch_size, len_seq, state_len)

        Return:
            - policy: (torch.Tensor): policy with shape (batch_size,num_action)
            - value: (torch.Tensor): value with shape (batch_size,1)
        """
        
        out        = self.fc(state)
        out,memory = self.transformer(out,memory,memory_mask,memory_indices)
        out        = out.squeeze(1)
        policy     = self.policy(out)
        #value      = self.value(out)

        return policy#, value, memory