import torch
import torch.nn.functional as F
import numpy as np
from numba import njit
from torch.distributions import Categorical

from models.rollout_buffer import RolloutBuffer
from models.distribution import Distribution
class Agent():
    """Agent"""
    def __init__(self,env,model,config):
        super().__init__()
        self.env           = env
        self.model         = model
        self.memory_length = config["transformer"]["memory_length"]
        self.num_blocks    = config["transformer"]["num_blocks"]
        self.embed_dim     = config["transformer"]["embed_dim"]
        self.reward        = config["rewards"]
        self.rollout       = RolloutBuffer(config,env.getStateSize(),env.getActionSize())
        self.dist          = Distribution()

    @torch.no_grad()
    def play(self,state,per):
        """
        Overview:
            Agent's play function

        Arguments:
            - state: (`np.array`): state
            - per: (`List`): per file

        Returns:
            - action: (`int`): Agent's action
            - per: (`List`): per file
        """
        list_action     = self.env.getValidActions(state)
        action_mask     = torch.tensor(list_action,dtype=torch.float32)
        with torch.no_grad():
            try: # prevent out of index
                tensor_state        = torch.tensor(state.reshape(1,-1),dtype=torch.float32)
                sliced_memory       = self.rollout.batched_index_select(
                    self.rollout.batch["memory"][self.rollout.game_count],
                    0,
                    self.rollout.memory_indices_batch[0][self.rollout.step_count])
                policy,value,memory = self.model(
                    tensor_state,
                    sliced_memory.unsqueeze(0),
                    self.rollout.memory_mask_batch[self.rollout.game_count,self.rollout.step_count].unsqueeze(0),
                    self.rollout.memory_indices_batch[self.rollout.game_count,self.rollout.step_count].unsqueeze(0))
                policy          = policy.squeeze()
                
                action,log_prob = self.dist.sample_action(policy,action_mask)
            

            
                if self.env.getReward(state)==-1:
                    self.rollout.add_data(state      = torch.from_numpy(state),
                                        action       = action,
                                        value        = value.item(),
                                        reward       = 0.0,
                                        done         = 0,
                                        valid_action = action_mask,
                                        prob         = log_prob,
                                        memory       = memory.squeeze(0),
                                        policy       = policy
                                        )
                    self.rollout.step_count+=1
                else:
                    self.rollout.add_data(state      = torch.from_numpy(state),
                                        action       = action, 
                                        value        = value.item(),
                                        reward       = self.reward[int(self.env.getReward(state))] * 1.0,
                                        done         = 1,
                                        valid_action = action_mask,
                                        prob         = log_prob,
                                        memory       = memory.squeeze(0),
                                        policy       = policy
                                        )
                    self.rollout.game_count+=1
                    self.rollout.step_count=0
            except:
                action      = np.random.choice(np.where(list_action==1)[0])

            if action_mask[action] != 1:
                action      = np.random.choice(np.where(list_action==1)[0])
        
        return action,per 
    
    def run(self,num_games:int)->float:
        """
        Overview:
            Run custom environment and return win rate.

        Arguments:
            - num_games: (`int`): number of games.
            
        """
        win_rate =  self.env.run(self.play,num_games,np.array([0.]),1)[0] / num_games
        # print(num_games,win_rate)
        return win_rate
    
    @njit()
    def bot_max_eps_length(self,state, perData):
        validActions = self.env.getValidActions(state)
        arr_action = np.where(validActions == 1)[0]
        idx = np.random.randint(0, arr_action.shape[0])
        perData[0]+=1
        if self.env.getReward(state)!=-1:
            if perData[0] > perData[1]:
                perData[1] = perData[0]
            perData[0] = 0
        return arr_action[idx], perData
    