import torch.nn as nn
from model.MultiheadAttention import MultiHeadAttention
from model.FeedForward import FeedFoward
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, block_size, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        self.sa = MultiHeadAttention(n_head, n_embd, block_size, 0.0)
        self.ffwd = FeedFoward(n_embd, 0.0)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x