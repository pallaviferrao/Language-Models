from Tokenize.tokenize import Corpus
import torch
import numpy as np
from model.rnn_simple import  VanillaRNN
import random
import time

#char RNN from scratch
def encode_data(X):
        """
        Encodes input text to integers
        Input:
            X - input text (string)
        Outputs:
            X_encoded - list with encoded characters from input text
        """
        X_encoded = []

        for char in X:
            X_encoded.append(char_to_idx[char])
        return X_encoded


def prepare_batches(X, index,seq_length,vocab_size):
        """
        Prepares one-hot-encoded X and Y batches that will be fed into the RNN
        Input:
            X - encoded input data (string)
            index - index of batch training loop, used to create training batches
        Output:
            X_batch - one-hot-encoded list
            y_batch - similar to X_batch, but every character is shifted one time step to the right
        """
        X_batch_encoded = X[index: index + seq_length]
        y_batch_encoded = X[index + 1: index + seq_length + 1]

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

        return X_batch, y_batch


def split_text_train_test(text, train_ratio=0.8):
  """Splits text into training and testing sets.

  Args:
    text: The text to split.
    train_ratio: The proportion of the text to use for training.

  Returns:
    A tuple containing the training and testing text.
  """
  lines = text.splitlines()
  random.shuffle(lines)  # Shuffle the lines randomly
  split_index = int(len(lines) * train_ratio)
  train_text = '\n'.join(lines[:split_index])
  test_text = '\n'.join(lines[split_index:])
  return train_text, test_text

embed_size = 128    # Embedding layer size , input to the LSTM
hidden_size = 75  # Hidden size of LSTM units
num_layers = 2      # no LSTMs stacked
num_epochs = 50    # total no of epochs
batch_size = 128     # batch size
seq_length = 100    # sequence length
learning_rate =0.000001 # learning rate

print("test1", torch.randint(0, 1, (1,)).long().unsqueeze(1))

file_content = None
with open('data/compliments.txt', 'r') as file:
    file_content = file.read()

# What is the vocabulary size ?
train_text, test_text = split_text_train_test(file_content)
chars = set(train_text)
print("Number of unique characters", chars)
vocab_size = len(chars)
print("Vocab", vocab_size)
smooth_loss = -np.log(1.0 / vocab_size) * seq_length
char_to_idx = {w: i for i,w in enumerate(chars)}
idx_to_char = {i: w for i,w in enumerate(chars)}
model = VanillaRNN(char_to_idx, idx_to_char, vocab_size, epochs = num_epochs, seq_len =seq_length )
# J, params = model.train(train_text)
J = []
epsilon = 1e-12
X_encoded = encode_data(train_text)
print("Dataset Length", len(X_encoded))
num_batches = len(X_encoded) //batch_size  # Num od sentence to divife it to
X_encoded = X_encoded[:num_batches*batch_size]
for i in range(num_epochs):
    epoch_start = time.time()
    # batch_start = time.time()
    # for k in range(0,len(X_encoded) - batch_size,batch_size):
    #     batch_start = time.time()
    #     X_batch_encoded = X_encoded[k:k+batch_size]
    for j in range(0, len(X_encoded)- seq_length, seq_length):
            X_batch, y_batch = prepare_batches(X_encoded,j,seq_length, vocab_size)  # fetch words for one seq length
            y_pred, h = model.forward_pass(X_batch)
            loss = 0
            model.clip_grads()
            for t in range(seq_length):
              # print(y_pred[t][0][np.argmax(y_batch[t])])
              prob = max(y_pred[t][0][np.argmax(y_batch[t])], epsilon)
              loss += -np.log(prob)
               # loss += -np.log(y_pred[t][0][np.argmax(y_batch[t])])
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            J.append(smooth_loss)
            model.backward_pass(X_batch, y_batch, y_pred, h)
            # model.update_params()

            batch_num = i * num_epochs + j / seq_length + 1
            model.update_params_adam(batch_num)

            if(j%40000 ==0):
                s_index = char_to_idx[' ']
                s, logVal = model.sample_pass(sample_size=100, start_index=s_index)
                length_s = len(s)
                strp = " "
                for ind in s:
                    word = idx_to_char[ind.item()]
                    strp = strp + word
                print(strp, "\n")
                logVal = -1 / length_s * (logVal)
                perplexity = np.power(2, logVal)
                print("perplexity", perplexity)

        # batch_end = time.time() - batch_start
        # print('Time to update ', batch_size, ' sentences is ', batch_end)
    epoch_end = time.time() - epoch_start
    print('Time to update ', epoch_end, ' epoch ', epoch_end)
    print('Epoch:', i + 1, "\tLoss:", loss, "")
    s_index = char_to_idx[' ']
    s, logVal = model.sample_pass(sample_size=100, start_index=s_index)
    length_s = len(s)
    strp = " "
    for ind in s:
        word = idx_to_char[ind.item()]
        strp = strp + word
    print(strp, "\n")
    logVal = -1 / length_s * (logVal)
    perplexity = np.power(2, logVal)
    print("perplexity", perplexity)

