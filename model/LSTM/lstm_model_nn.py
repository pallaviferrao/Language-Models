import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        #Create an embedding for each word
        self.embed = nn.Embedding(vocab_size, embed_size)  # maps words to feature vectors
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM layer
        self.linear = nn.Linear(hidden_size, vocab_size)  # Fully connected layer

    def forward(self, x, h):
        # Perform Word Embedding
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)  # (input , hidden state)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)
