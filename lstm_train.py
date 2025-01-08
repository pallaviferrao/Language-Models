import random

from model.LSTM.lstm_from_scratch import LSTM
from Tokenize.tokenizer import Tokenize
import torch
import numpy as np
import time

#Word LSTM from scratch
class Dictionary(object):
  def __init__(self):
    self.word2idx = {}
    self.idx2word = {}
    self.idx = 0

  def add_word(self, word):
    if word not in self.word2idx:
      self.word2idx[word] = self.idx
      self.idx2word[self.idx] = word
      self.idx += 1

  def __len__(self):
    return len(self.word2idx)


class Corpus(object):

  def __init__(self):
    self.dictionary = Dictionary()

  def get_data(self, path, batch_size=20):
    with open(path, 'r') as f:
      tokens = 0
      for line in f:
        words = line.split() + ['<eos>']
        tokens += len(words)
        for word in words:
          self.dictionary.add_word(word)
          # Create a 1-D tensor which contains index of all the words in the file with the help of word2idx
    ids = torch.LongTensor(tokens)
    token = 0
    with open(path, 'r') as f:
      for line in f:
        words = line.split() + ['<eos>']
        for word in words:
          ids[token] = self.dictionary.word2idx[word]
          token += 1
    # no of required batches
    num_batches = ids.shape[0] // batch_size
    # Remove the remainder from the last batch , so that always batch size is constant
    ids = ids[:num_batches * batch_size]
    # return (batch_size,num_batches)
    ids = ids.view(batch_size, -1)
    return ids

  def get_word(self, idx):
      return self.dictionary.idx2word[idx]



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

tokenize = Tokenize()
file_content = None
with open('data/compliments.txt', 'r') as file:
    file_content = file.read()


embed_size = 128    # Embedding layer size , input to the LSTM
hidden_size = 75  # Hidden size of LSTM units
num_layers = 1      # no LSTMs stacked
num_epochs = 50    # total no of epochs
batch_size = 128     # batch size
seq_length = 100     # sequence length
learning_rate = 0.005 # learning rate
corpus = Corpus()

ids = corpus.get_data('data/compliments.txt',batch_size)
# Assuming 'file_content' is defined as in the previous code
train_text, test_text = split_text_train_test(file_content)
# id2 = tokenize.corpus_create('data/compliments.txt')
# Now you have 'train_text' and 'test_text'
print("Train text (first 200 chars):", train_text[:200])
print("Test text (first 200 chars):", test_text[:200])
chars = set(train_text)
vocab_size = len(chars)
vocab_size = len(corpus.dictionary)
print(vocab_size)

print('data has %d characters, %d unique' % (len(train_text), vocab_size))
# creating dictionaries for mapping chars to ints and vice versa
char_to_idx = {w: i for i,w in enumerate(chars)}
idx_to_char = {i: w for i,w in enumerate(chars)}

char_to_idx = tokenize.wordToId
idx_to_char = tokenize.IdToWord
model = LSTM(char_to_idx, idx_to_char, vocab_size, epochs = 50)

# J, params = model.train(train_text, ids)

epochs = 50
n_h = 100
def train1( X, verbose=True):
    """
    Main method of the LSTM class where training takes place
    """
    J = []  # to store losses
    smooth_loss = 0
    print("HEre")
    num_batches = len(X) // seq_length # trim input to have full sequences

    for epoch in range(epochs):
        epoch_start = time.time()
        h_prev = np.zeros((n_h, 1))
        c_prev = np.zeros((n_h, 1))

        for k in range(batch_size):
            batch_time = time.time()
            for j in range(0, len(ids) - seq_length, seq_length):
                # prepare batches
                x_batch = ids[k, j:j + seq_length]  # fetch words for one seq length
                y_batch = ids[k, (j + 1):(j + 1) + seq_length]  # shifted by one word from inputs
                # x_batch = [self.char_to_idx[ch] for ch in X_trimmed[j: j + self.seq_len]]
                # y_batch = [self.char_to_idx[ch] for ch in X_trimmed[j + 1: j + self.seq_len + 1]]

                loss, h_prev, c_prev = model.forward_backward(x_batch, y_batch, h_prev, c_prev)

                # smooth out loss and store in list
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
                J.append(smooth_loss)

                # check gradients
                if epoch == 0 and k == 0:
                    model.gradient_check(x_batch, y_batch, h_prev, c_prev, num_checks=10, delta=1e-7)

                model.clip_grads()

                batch_num = epoch * epochs + j / seq_length + 1
                model.update_params(batch_num)

                # print out loss and sample string
            print("Batch time", time.time() - batch_time)

        print('Epoch:', epoch, 'epoch time', time.time()-epoch_start, '\tBatch:', j, "-", j + seq_length,
              '\tLoss:', round(smooth_loss, 2))
        s,logVal = model.sample_pass(h_prev, c_prev, sample_size=250)
        length_s = len(s)
        strp = " "
        for ind in s:
            word = corpus.get_word(ind)
            strp = strp+  " "+ word
        print(strp, "\n")
        logVal = -1 / length_s * (logVal)
        perplexity = np.power(2, logVal)
        print("perplexity", perplexity)


    # return J, self.params

train1(train_text)