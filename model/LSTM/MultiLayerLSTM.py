from torch import nn, optim
import torch
from model.LSTM.LSTMModel import LSTMCell
class MultiLayerLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(MultiLayerLSTM, self).__init__()
        self.input_dim, self.hidden_dim, self.num_layers = input_dim, hidden_dim, num_layers
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(LSTMCell(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, input_dim)
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x, h):
        # x has shape [batch_size, seq_len, embed_dim]
        # h is a tuple containing h and c, each have shape [layer_num, batch_size, hidden_dim]
        hidden, cell = h
        output_hidden, output_cell = self.layers[0](x, hidden[0], cell[0])
        new_hidden, new_cell = [output_hidden[:, -1].unsqueeze(0)], [output_cell[:, -1].unsqueeze(0)]
        for i in range(1, self.num_layers):
            output_hidden, output_cell = self.layers[i](self.dropout(output_hidden), hidden[i], cell[i])
            new_hidden.append(output_hidden[:, -1].unsqueeze(0))
            new_cell.append(output_cell[:, -1].unsqueeze(0))
        return self.linear(self.dropout(output_hidden)), (torch.concat(new_hidden, dim=0), torch.concat(new_cell, dim=0))