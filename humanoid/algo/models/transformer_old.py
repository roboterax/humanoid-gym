import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *
from .gru import *

class SinusoidalPE(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        super().__init__()
        freqs     = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        """
        Overview:
            Compute positional embedding

        Arguments:
            - seq_len: (`int`): sequence length.

        Returns:
            - pos_embedding: (:obj:`torch.Tensor`): positional embedding. Shape (seq_len, 1, embedding_dim)
        """
        seq            = torch.arange(int(seq_len) - 1, -1, -1.)
        sinusoidal_inp = seq.view(-1,1) * self.inv_freqs.view(1,-1)
        pos_emb        = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, config):
        """
        Overview: Initialize a Transformer Block.
        
        Arguments:
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            config (dict): Configuration parameters for the GRU gate.
        """
        super().__init__()
        self.attention   = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.gate1       = GRUGate(embed_dim, config["transformer"]['gru_bias'])
        self.gate2       = GRUGate(embed_dim, config["transformer"]['gru_bias'])

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

    def forward(self, query, key, mask=None):
        """
        Overview: Forward pass of the Transformer Block.
        
        Arguments:
            query (tensor): Query tensor.
            key (tensor): Key tensor.
            mask (tensor): Mask tensor for attention, indicating which elements to attend to.
        
        Returns:
            out (tensor): Output tensor after the Transformer Block.
        """
        norm_key = self.layer_norm1(key)
        Y        = self.attention(self.layer_norm1(query), norm_key, norm_key, mask)
        out      = self.gate1(query, Y)
        E        = self.fc(self.layer_norm2(out))
        out      = self.gate2(out, E)
        assert torch.isnan(out).any() == False, "Transformer block returned NaN!"

        return out





class GatedTransformerXL(nn.Module):
    """
    Overview:
        Initialize a Gated Transformer XL model.
    
    Arguments:
        config (dict): Configuration parameters for the model.
        input_dim (int): Dimensionality of the input.
        max_episode_steps (int): Maximum number of episode steps.
    """
    def __init__(self, 
                 config:dict,
                 input_dim:int,
                 max_episode_steps=500) -> None:
        
        super().__init__()
        self.config            = config
        self.num_blocks        = config["transformer"]["num_blocks"]
        self.embed_dim         = config["transformer"]["embed_dim"]
        self.num_heads         = config["transformer"]["num_heads"]
        self.heads_dim         = self.embed_dim // self.num_heads
        self.memory_length     = config["transformer"]["memory_length"]
        self.max_episode_steps = max_episode_steps
        self.activation        = nn.GELU()
        # Input embedding layer
        self.linear_embedding  = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))
        self.pos_embedding     = SinusoidalPE(dim = self.embed_dim)(self.max_episode_steps)
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config) 
            for _ in range(self.num_blocks)])
        
    def forward(self, h, memories, mask, memory_indices):
        """
        Arguments:
            h {torch.tensor} -- Input (query)
            memories {torch.tesnor} -- Whole episoded memories of shape (N, L, num blocks, D)
            mask {torch.tensor} -- Attention mask (dtype: bool) of shape (N, L)
            memory_indices {torch.tensor} -- Memory window indices (dtype: long) of shape (N, L)
        Returns:
            {torch.tensor} -- Output of the entire transformer encoder
            {torch.tensor} -- Out memories (i.e. inputs to the transformer blocks)
        """
        # Feed embedding layer and activate
        h = self.activation(self.linear_embedding(h))

        # Add positional encoding to every transformer block input
        #pos_embedding = self.pos_embedding[memory_indices.long()]
        #memories = memories + pos_embedding.unsqueeze(2)
        # Forward transformer blocks
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            #h = block(h.unsqueeze(1), memories[:, :, i], mask) # args:  query, key, mask
            h = block(h, h, mask)
            h = h.squeeze()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
        return h, torch.stack(out_memories, dim=1)