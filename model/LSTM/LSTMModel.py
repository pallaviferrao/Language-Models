import torch.nn.functional as F
from torch import nn, optim
import torch

class LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.forget_input, self.forget_hidden, self.forget_bias = self.create_gate_parameters()
        self.input_input, self.input_hidden, self.input_bias = self.create_gate_parameters()
        self.output_input, self.output_hidden, self.output_bias = self.create_gate_parameters()
        self.cell_input, self.cell_hidden, self.cell_bias = self.create_gate_parameters()

    def create_gate_parameters(self):
        input_weights = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
        hidden_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(input_weights)
        nn.init.xavier_uniform_(hidden_weights)
        bias = nn.Parameter(torch.zeros(self.hidden_dim))
        return input_weights, hidden_weights, bias

    def forward(self, x, h, c):
        # x has shape [batch_size, seq_len, input_size]
        output_hiddens, output_cells = [], []
        for i in range(x.shape[1]):
            # print("x", x[:, i].shape)
            forget_gate = F.sigmoid((x[:, i] @ self.forget_input) + (h @ self.forget_hidden) + self.forget_bias)
            input_gate = F.sigmoid((x[:, i] @ self.input_input) + (h @ self.input_hidden) + self.input_bias)
            output_gate = F.sigmoid((x[:, i] @ self.output_input) + (h @ self.output_hidden) + self.output_bias)
            input_activations = F.tanh((x[:, i] @ self.cell_input) + (h @ self.cell_hidden) + self.cell_bias)
            c = (forget_gate * c) + (input_gate * input_activations)
            h = F.tanh(c) * output_gate
            squeeze_h = h.unsqueeze(1)
            output_hiddens.append(squeeze_h)
            squeeze_c = c.unsqueeze(1)
            output_cells.append(squeeze_c)
        return torch.concat(output_hiddens, dim=1), torch.concat(output_cells, dim=1)