import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
from utils import one_hot
from utils import sample
from torch.utils.data import DataLoader

from torch import nn, optim
from tqdm import tqdm

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
            # forget_gate = F.sigmoid((x[:, i] @ self.forget_input) + (h @ self.forget_hidden) + self.forget_bias)
            h = F.tanh((x[:, i] @ self.input_input) + (h @ self.input_hidden) + self.input_bias)
            # output_gate = F.sigmoid((x[:, i] @ self.output_input) + (h @ self.output_hidden) + self.output_bias)
            # input_activations = F.tanh((x[:, i] @ self.cell_input) + (h @ self.cell_hidden) + self.cell_bias)
            # # c = (forget_gate * c) + (input_gate * input_activations)
            # # h = F.tanh(c) * output_gate
            #
            # output_hiddens.append(h.unsqueeze(1))
            # output_cells.append(c.unsqueeze(1))
            m = nn.Softmax(dim=1)
            output_hidden =  h@self.hidden_output  + self.bias_output
            # print(output_hidden.shape)
            # print("Output hidden", self.hidden_output.shape)
            output_hiddens.append(h.unsqueeze(1))
            abc = output_hidden.unsqueeze(1)
            output_cells.append(abc)

        # print(torch.concat(output_cells, dim=1).shape)
        return torch.cat(output_hiddens, dim=1), torch.cat(output_cells, dim=1)



class MultiLayerLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(MultiLayerLSTM, self).__init__()
        self.input_dim, self.hidden_dim, self.num_layers = input_dim, hidden_dim, num_layers
        self.layers = nn.ModuleList()
        self.layers.append(RNNPytorch(input_dim, hidden_dim, input_dim))
        for _ in range(num_layers - 1):
            self.layers.append(RNNPytorch(hidden_dim, hidden_dim, output_dim=input_dim))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, input_dim)

        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.ln = nn.LayerNorm(hidden_dim)


    def forward(self, x, h):
        # x has shape [batch_size, seq_len, embed_dim]
        # h is a tuple containing h and c, each have shape [layer_num, batch_size, hidden_dim]
        hidden, cell = h
        output_hidden, output_cell = self.layers[0](x, hidden[0], cell[0])

        new_hidden, new_cell = [output_hidden[:, -1].unsqueeze(0)], [output_cell.unsqueeze(0)]
        for i in range(1, self.num_layers):
            output_hidden, output_cell = self.layers[i](self.dropout(output_hidden), hidden[i], cell[i])

            new_hidden.append(output_hidden[:, -1].unsqueeze(0))
            new_cell.append(output_cell.unsqueeze(0))

        return self.linear(self.dropout(self.ln(output_hidden))), (torch.cat(new_hidden, dim=0), torch.cat(new_cell, dim=0))



class SequenceDataset(Dataset):
    def __init__(self, txt_path: str, seq_length: int = 100) -> None:
        f = open(txt_path, encoding="utf-8")
        self.seq_length = seq_length
        self.corpus = "".join(f.readlines()).lower()
        self.vocab = sorted(list(set(self.corpus)))
        self.vocab_to_idx = {i: idx for (idx, i) in enumerate(self.vocab)}
        self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}
        self.corpus_encoded = torch.zeros(len(self.corpus), len(self.vocab))

        for i, char in tqdm(enumerate(self.corpus), total=len(self.corpus), position=0, leave=True):
            self.corpus_encoded[i] = one_hot(self.vocab_to_idx[char], len(self.vocab))

        self.xvals = self.corpus_encoded[:-1]
        self.yvals = self.corpus_encoded[1:]

        extra_vals = len(self.xvals) % seq_length

        self.xvals = self.xvals[:-extra_vals].view(len(self.xvals) // seq_length, seq_length, len(self.vocab))
        self.yvals = self.yvals[:-extra_vals].view(len(self.yvals) // seq_length, seq_length, len(self.vocab))


    def __len__(self) -> int:
        return self.xvals.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xvals[idx], self.yvals[idx]


SEQ_LENGTH = 100
HIDDEN_SIZE = 512
NUM_LAYERS = 2

DROPOUT = 0.5

LR = 0.005
BATCH_SIZE = 128
EPOCHS = 100

dataset = SequenceDataset("../../data/compliments.txt", seq_length=SEQ_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = MultiLayerLSTM(len(dataset.vocab), HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
opt = optim.Adam(model.parameters(), lr = LR)
# opt = optim.Adagrad(model.parameters(), lr = LR)
crit = nn.CrossEntropyLoss()


for e in range(1, EPOCHS + 1):
    loop = tqdm(loader, total=len(loader), leave=True, position=0)
    loop.set_description(f"Epoch : [{e}/{EPOCHS}] | ")
    total_loss = 0
    total_len = 0
    for x, y in loop:
        opt.zero_grad()
        h = (torch.zeros((NUM_LAYERS, x.shape[0], HIDDEN_SIZE)), torch.zeros((NUM_LAYERS, x.shape[0], HIDDEN_SIZE)))
        yhat, h = model.forward(x, h)

        loss = crit(yhat.view(-1, yhat.shape[-1]), y.view(-1, y.shape[-1]))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()

        total_loss += loss.item()
        total_len += 1
        loop.set_postfix(average_loss = total_loss / total_len)

    if e % 2 == 0:
        print(f"\n{'=' * 50}\nSample output: \n{sample(model, dataset, 'thou', HIDDEN_SIZE, 400, NUM_LAYERS, )}\n{'=' * 50}\n")
        torch.save(model.state_dict(), "lstm-weights.pth")