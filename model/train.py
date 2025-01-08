import random
from model.RNN.rnn_simple import VanillaRNN
from Tokenize.tokenizer import Tokenize
import torch

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
with open('compliments.txt', 'r') as file:
    file_content = file.read()


embed_size = 128    # Embedding layer size , input to the LSTM
hidden_size = 1024  # Hidden size of LSTM units
num_layers = 1      # no LSTMs stacked
num_epochs = 10     # total no of epochs
batch_size = 75     # batch size
seq_length = 100     # sequence length
learning_rate = 0.002 # learning rate
corpus = Corpus()

ids = corpus.get_data('compliments.txt',batch_size)
# Assuming 'file_content' is defined as in the previous code
train_text, test_text = split_text_train_test(file_content)
id2 = tokenize.corpus_create('compliments.txt')
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

# char_to_idx = tokenize.wordToId
# idx_to_char = tokenize.IdToWord
model = VanillaRNN(tokenize.wordToId, tokenize.IdToWord, vocab_size, epochs = 10)
# model = VanillaRNN(char_to_idx, idx_to_char, vocab_size, epochs = 10)

J, params = model.train(train_text, ids)