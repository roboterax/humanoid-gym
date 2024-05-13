import torch
import torch.nn as nn
from torch.distributions import Categorical, kl_divergence

class Distribution():
    def sample_action(self, policy, action_mask):
        """
        Overview: 
            Sample an action from a given policy distribution, considering an action mask.
        
        Arguments:
            policy (tensor): Logits of the policy distribution.
            action_mask (tensor): Mask indicating which actions are valid (1) or invalid (0).
        
        Returns:
            action (int): Sampled action from the distribution.
            log_prob (tensor): Log probability of the sampled action.
        """
        distribution = Categorical(logits=policy.masked_fill(action_mask == 0, float("-1e20")))
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
        
    def log_prob(self, policy, action, action_mask):
        """
        Overview: 
            Calculate the log probability and entropy of a given action in a policy distribution, considering an action mask.
        
        Arguments:
            policy (tensor): Logits of the policy distribution.
            action (int): Action for which log probability and entropy are computed.
            action_mask (tensor): Mask indicating which actions are valid (1) or invalid (0).
        
        Returns:
            log_prob (tensor): Log probability of the given action.
            entropy (tensor): Entropy of the policy distribution.
        """
        distribution = Categorical(logits=policy.masked_fill(action_mask == 0, float("-1e20")))
        return distribution.log_prob(action), distribution.entropy()

    def kl_divergence(self, policy, policy_new):
        """
        Overview: Calculate the Kullback-Leibler (KL) divergence between two policy distributions.
        
        Arguments:
            policy (tensor): Logits of the first policy distribution.
            policy_new (tensor): Logits of the second policy distribution.
        
        Returns:
            kl_divergence (tensor): KL divergence between the two policy distributions.
        """
        return kl_divergence(Categorical(logits=policy), Categorical(logits=policy_new))

    
