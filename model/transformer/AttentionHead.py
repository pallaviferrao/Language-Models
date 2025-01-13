import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    """ nn.Module is required to reqister memory and other nn functions."""
    def __init__(self, head_size, n_embd,block_size,dropout ):
        """ one head of self-attention
        We need Key, Query and Value
        x dimension is block/sequence length * batchsize * embed
        k,q and v deal with the embeddings so the shape would be embed * output size/headsize
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        block_size, batch_size, embed_size = x.shape
        k = self.key(x)
        q = self.query(x)# (B,T,C)
        weights = q @ k.transpose(-2, -1) # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights  = weights/math.sqrt(embed_size)
        weights = weights.masked_fill(self.tril[:batch_size, :batch_size] == 0, float('-inf'))  # (B, T, T)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)  # (B,T,C)
        out = weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd,block_size,dropout ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out