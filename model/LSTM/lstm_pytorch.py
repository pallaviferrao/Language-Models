import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
from utils import one_hot
from utils import sample
from torch.utils.data import DataLoader

from torch import nn, optim
from tqdm import tqdm

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
            # print("output cells", len(output_cells))
            # print("output cells", torch.concat(output_cells, dim=1).shape)
        return torch.concat(output_hiddens, dim=1), torch.concat(output_cells, dim=1)



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
NUM_LAYERS = 1
DROPOUT = 0.5

LR = 0.005
BATCH_SIZE = 128
EPOCHS = 2

dataset = SequenceDataset("../../data/compliments.txt", seq_length=SEQ_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(len(dataset.vocab))
model = MultiLayerLSTM(len(dataset.vocab), HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
opt = optim.Adam(model.parameters(), lr = LR)
crit = nn.CrossEntropyLoss()


for e in range(1, EPOCHS + 1):
    loop = tqdm(loader, total=len(loader), leave=True, position=0)
    loop.set_description(f"Epoch : [{e}/{EPOCHS}] | ")
    total_loss = 0
    total_len = 0

    for x, y in loop:
        print(x.shape, y.shape)
        opt.zero_grad()
        h = (torch.zeros((NUM_LAYERS, x.shape[0], HIDDEN_SIZE)), torch.zeros((NUM_LAYERS, x.shape[0], HIDDEN_SIZE)))
        yhat, h = model.forward(x, h)
        print(yhat[0][0])
        print(y[0][0])
        # print("Testing", h[0].shape)
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