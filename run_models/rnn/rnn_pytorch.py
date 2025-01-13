import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
from model.utils import one_hot, sample
from torch.utils.data import DataLoader
import time
from torch import nn, optim
from tqdm import tqdm
from model.DataProcess import Sequential
from model.RNN.MultiLayerRNN import MultiLayerRNN

SEQ_LENGTH = 100
HIDDEN_SIZE = 512
NUM_LAYERS = 3

DROPOUT = 0.5

LR = 0.00005
BATCH_SIZE = 128
EPOCHS = 100

dataset = SequenceDataset("../../data/compliments.txt", seq_length=SEQ_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = MultiLayerRNN(len(dataset.vocab), HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
opt = optim.Adam(model.parameters(), lr = LR)
# opt = optim.Adagrad(model.parameters(), lr = LR)
crit = nn.CrossEntropyLoss()


for e in range(1, EPOCHS + 1):
    epoch_start = time.time()
    loop = tqdm(loader, total=len(loader), leave=True, position=0)
    loop.set_description(f"Epoch : [{e}/{EPOCHS}] | ")
    total_loss = 0
    total_len = 0
    for x, y in loop:

        opt.zero_grad()
        h = (torch.zeros((NUM_LAYERS, x.shape[0], HIDDEN_SIZE)), torch.zeros((NUM_LAYERS, x.shape[0], HIDDEN_SIZE)))
        yhat, h = model.forward(x, h)
        # print(yhat.view(-1, yhat.shape[-1]).shape)
        loss = crit(yhat.view(-1, yhat.shape[-1]), y.view(-1, y.shape[-1]))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()

        total_loss += loss.item()
        total_len += 1
        loop.set_postfix(average_loss = total_loss / total_len)
    epoch_end = time.time() - epoch_start
    if e % 2 == 0:
        print(f"\n{'=' * 50}\nSample output: \n{sample(model, dataset, ' ', HIDDEN_SIZE, 400, NUM_LAYERS, )}\n{'=' * 50}\nEpoch Time: {epoch_end}")
        torch.save(model.state_dict(), "lstm-weights.pth")