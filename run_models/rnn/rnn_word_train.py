from Tokenize.tokenize import Corpus
import torch
import numpy as np
from model.RNN.rnn import  VanillaRNN
from Tokenize.tokenizer import Tokenize
import time

#Word RNN from scratch

def prepare_batches(X_batch_encoded, y_batch_encoded, vocab_size):
    """
    Prepares one-hot-encoded X and Y batches that will be fed into the RNN
    Input:
        X - encoded input data (string)
        index - index of batch training loop, used to create training batches
    Output:
        X_batch - one-hot-encoded list
        y_batch - similar to X_batch, but every character is shifted one time step to the right
    """
    # X_batch_encoded = X[index: index + seq_len]
    # y_batch_encoded = X[index + 1: index + seq_len + 1]

    X_batch = []
    y_batch = []

    for i in X_batch_encoded:
        one_hot_char = np.zeros((1, vocab_size))
        one_hot_char[0][i] = 1
        X_batch.append(one_hot_char)

    for j in y_batch_encoded:
        one_hot_char = np.zeros((1, vocab_size))
        one_hot_char[0][j] = 1
        y_batch.append(one_hot_char)
    # print(X_batch.shape)
    return X_batch, y_batch


embed_size = 128    # Embedding layer size , input to the LSTM
hidden_size = 75  # Hidden size of LSTM units
num_layers = 2      # no LSTMs stacked
num_epochs = 10    # total no of epochs
batch_size = 128     # batch size
seq_length = 20    # sequence length
learning_rate = 0.02 # learning rate
corpus = Corpus()
ids = corpus.get_data('./data/compliments.txt',batch_size)
tokenize = Tokenize()


print("id",corpus.get_id('<sos>'))
print("test1", torch.randint(0, 1, (1,)).long().unsqueeze(1))
# ids tensors contain all the index of each words
print(ids.shape)


# What is the vocabulary size ?
vocab_size = len(corpus.dictionary)
print(vocab_size)
epochs = 50
model = VanillaRNN(tokenize.wordToId, tokenize.IdToWord, vocab_size, epochs = 50, seq_len =seq_length )
smooth_loss = -np.log(1.0 / vocab_size) * seq_length
J = []
print("Size",ids.size(1))
num_batches = len(ids) //seq_length  # Num od sentence to divife it to
# ids = ids[:num_batches * seq_length]
start_time = time.time()
for i in range(epochs):
    epoch_start = time.time()
    for k in range(batch_size):
        batch_start = time.time()
        for j in range(0, ids.size(1) - seq_length, seq_length):

            # X_batch, y_batch = prepare_batches(ids, j, seq_length, vocab_size)
            X_batch_encoded = ids[k, j:j + seq_length]  # fetch words for one seq length
            y_batch_encoded = ids[k, (j + 1):(j + 1) + seq_length]  # shifted by one word from inputs
            X_batch, y_batch = prepare_batches(X_batch_encoded, y_batch_encoded, vocab_size)
            # print("X shape",X_batch.shape)
            y_pred, h = model.forward_pass(X_batch)
            loss = 0
            for t in range(seq_length):
               loss += -np.log(y_pred[t][0][np.argmax(y_batch[t])])
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            J.append(smooth_loss)

            model.backward_pass(X_batch, y_batch, y_pred, h)
            # print("Backward done done")
            model.update_params()

        batch_end = time.time() - batch_start
    epoch_end = time.time() - epoch_start
    print('Epoch:', i + 1, "\tLoss:", loss, "", 'epoch time', epoch_end)
    index1 = corpus.get_id("Your")
    s, logVal = model.sample_pass(sample_size=20, start_index=index1)
    print(len(s))
    strp = " "
    for ind in s:
        word = corpus.get_word_for_id(ind.item())
        strp = strp + word + " "
    print(strp, "\n")
    logVal = -1 / len(s) * (logVal)
    perplexity = torch.pow(2, logVal)
    print("perplexity", perplexity)



# return J, model.params

