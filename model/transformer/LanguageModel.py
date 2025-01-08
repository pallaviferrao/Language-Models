# super simple bigram model
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.AttentionBlock import Block
from model.PositionalEncoding import PositionalEncoding


class LanguageModel(nn.Module):
    '''
    This model is supposed to predict the next character or word using Blocks of Multihead attention.
    We first created a self attention head
    then we stack the attention head to create a multi attention head
    And then we stack the multi-attention  head block

    '''
    def __init__(self, device, vocab_size, n_embd, block_size, n_head, n_layer):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.device = device
        self.block_size = block_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.position_embedding_table = PositionalEncoding(n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.attention_blocks = nn.Sequential(*[Block(self.n_embd, self.block_size, n_head=self.n_head) for _ in range(self.n_layer)])



    def forward(self, x_seq, y=None):
        block_size, batch_size = x_seq.shape
        #We want to first create embeddings for each entry in x of size embed.
        # It maps the token value to the first dimension in the nn.Embedding which is why the first dimension is the vocab size.
        token_embedding = self.token_embeddings(x_seq)
        #after this we need to get the position of each value
        #TODO: This should be the sin and cos formula which we will add tomorrow
        #For now each element in the batch needs a position embedding

        #pos_emb = self.position_embedding_table(torch.arange(batch_size, device= self.device)) # (T,C)
        pos_emb = PositionalEncoding(self.n_embd, batch_size)(torch.arange(batch_size, device= self.device))
        #Create sequence for Blocks for given number of layers
        # print(token_embedding.shape)
        # print(pos_emb.shape)
        # pos_emb = pos_emb.expand(block_size, -1, -1)
        # print(token_embedding.shape)
        # print(pos_emb.shape)

        x = token_embedding + pos_emb
        x = self.attention_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = y.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        logVal = 0
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            probs = probs / torch.sum(probs)

            logVal += torch.log(probs[:, idx_next])
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        logVal = -1 / max_new_tokens * (logVal)
        perplexity = torch.pow(2, logVal)
        print("perplexity", perplexity)

        return idx


