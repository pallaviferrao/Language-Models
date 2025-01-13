from torch import nn
import torch

class RNNPytorch(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNPytorch, self).__init__()
        self.input_dim, self.hidden_dim,self.output_dim = input_dim, hidden_dim,output_dim
        # # self.forget_input, self.forget_hidden, self.forget_bias = self.create_gate_parameters()
        self.input_input, self.input_hidden, self.input_bias = self.create_gate_parameters()
        # self.output_input, self.output_hidden, self.output_bias = self.create_gate_parameters()
        # self.cell_input, self.cell_hidden, self.cell_bias = self.create_gate_parameters()
        self.hidden_output = nn.Parameter(torch.zeros(self.hidden_dim, self.output_dim))
        self.bias_output = nn.Parameter(torch.zeros(self.output_dim))
        nn.init.xavier_uniform_(self.hidden_output)


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
            h = F.tanh((x[:, i] @ self.input_input) + (h @ self.input_hidden) + self.input_bias)
            output_hidden =  h@self.hidden_output  + self.bias_output
            output_hiddens.append(h.unsqueeze(1))
            abc = output_hidden.unsqueeze(1)
            output_cells.append(abc)

        return torch.cat(output_hiddens, dim=1), torch.cat(output_cells, dim=1)
