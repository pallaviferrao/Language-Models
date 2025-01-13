import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
from model.utils import one_hot
from model.utils import sample
from torch.utils.data import DataLoader
from model.LSTM.MultiLayerLSTM import MultiLayerLSTM
from model.DataProcess.Sequential import SequenceDataset
from torch import nn, optim
from tqdm import tqdm


SEQ_LENGTH = 100
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.5

LR = 0.005
BATCH_SIZE = 128
EPOCHS = 50

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
        # print(x.shape, y.shape)
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