import torch
import torch.nn as nn
from model.AttentionHead import AttentionHead
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, n_embd, dropout):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.n_embd = n_embd
        self.num_heads = num_heads

    def forward(self, x):
        #We expect that each output would be the embeddings divided by the heads
        head_size = self.n_embd // self.num_heads
        #TODO: Whats the difference between ModuleList and Sequential?
        multiple_heads = nn.ModuleList([AttentionHead(head_size,  self.n_embd, 32, 0.2) for _ in range(self.num_heads)])
        #Concat the results from the attention heads
        #concat over the last dimension
        out = torch.cat([head(x) for head in multiple_heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

