import torch
import torch.nn.functional as F
import numpy as np

def one_hot(c: int, vocab_size: int):
    encoded = torch.zeros(vocab_size)
    encoded[c] = 1.0
    return encoded


# def predict(model, char, hidden, dataset) -> tuple[str, torch.Tensor]:
#     with torch.no_grad():
#         char_encoded = one_hot(dataset.vocab_to_idx[char], len(dataset.vocab)).unsqueeze(0)
#         next_char, new_hidden = model.forward(char_encoded.unsqueeze(0), hidden)
#         next_char, new_hidden = next_char.view(-1), new_hidden
#         probs = F.softmax(next_char, dim=0)
#         char = torch.multinomial(probs, 1)[0].item()
#     return dataset.idx_to_vocab[char], new_hidden

def predict(model, char, hidden, dataset):
    with torch.no_grad():
        char_encoded = one_hot(dataset.vocab_to_idx[char], len(dataset.vocab)).unsqueeze(0)
        next_char, new_hidden = model.forward(char_encoded.unsqueeze(0), hidden)
        next_char, new_hidden = next_char.view(-1), new_hidden
        probs = F.softmax(next_char, dim=0)
        char = torch.multinomial(probs, 1)[0].item()
    return dataset.idx_to_vocab[char], new_hidden,probs[char]
def sample(model, dataset, init_seq, hidden_size, seq_length, num_layers) -> str:
    with torch.no_grad():
        out = init_seq
        h = (torch.zeros(num_layers, hidden_size), torch.zeros(num_layers, hidden_size))
        x = torch.zeros(len(init_seq), len(dataset.vocab))
        for i, char in enumerate(init_seq):
            x[i] = one_hot(dataset.vocab_to_idx[char], len(dataset.vocab))
        preds, h = model.forward(x.unsqueeze(0), h)
        out += dataset.idx_to_vocab[preds[0][-1].argmax().item()]
        logVal =0
        for i in range(seq_length):
            c, h, prob = predict(model, out[-1], h, dataset)
            logVal += torch.log(prob)
            out += c
        return out, logVal/seq_length

def sigmoid( x):
    return 1 / (1 + np.exp(-x))

def softmax( x):
    e_x = np.exp(x - np.max(x))  # max(x) subtracted for numerical stability
    return e_x / np.sum(e_x)



SEQ_LENGTH = 100
HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT = 0.5

LR = 0.001
BATCH_SIZE = 128
EPOCHS = 1000
