import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence



class MinLSTMCell(nn.Module):
    def __init__(self, units, input_shape):
        super(MinLSTMCell, self).__init__()
        self.units = units
        self.input_shape = input_shape

        # Initialize the linear layers for the forget gate, input gate, and hidden state transformation
        self.linear_f = nn.Linear(self.input_shape, self.units)
        self.linear_i = nn.Linear(self.input_shape, self.units)
        self.linear_h = nn.Linear(self.input_shape, self.units)

    def forward(self, pre_h, x_t):
        """
        pre_h: (batch_size, units) - previous hidden state (h_prev)
        x_t: (batch_size, input_size) - input at time step t
        """

        # Forget gate: f_t = sigmoid(W_f * x_t)
        f_t = torch.sigmoid(self.linear_f(x_t))  # (batch_size, units)

        # Input gate: i_t = sigmoid(W_i * x_t)
        i_t = torch.sigmoid(self.linear_i(x_t))  # (batch_size, units)

        # Hidden state: tilde_h_t = W_h * x_t
        tilde_h_t = self.linear_h(x_t)  # (batch_size, units)

        # Normalize the gates
        sum_f_i = f_t + i_t
        f_prime_t = f_t / sum_f_i  # (batch_size, units)
        i_prime_t = i_t / sum_f_i  # (batch_size, units)

        # New hidden state: h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t
        h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t  # (batch_size, units)

        return h_t  # (batch_size, units)


class MinRNN(nn.Module):
    def __init__(self, units, embedding_size, vocab_size, input_length):
        super(MinRNN, self).__init__()
        self.input_length = input_length
        self.units = units

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = MinLSTMCell(units, embedding_size)
        self.classification_model = nn.Linear(5, vocab_size)


    def forward(self, sentence):
        """
        Args:
            sentence: (batch_size, input_length)

        output:
            (batch_size, 1)

        """
        batch_size = sentence.shape[0]

        # Initialize the hidden state, only the h needs to be initialized
        pre_h = torch.zeros(batch_size, self.units, device=sentence.device)

        # Pass the sentence through the embedding layer for the word vectors embeddings
        embedded_sentence = self.embedding(sentence)

        sequence_length = embedded_sentence.shape[1]

        # Pass the entire sequence through the LSTM + hidden_state
        for i in range(sequence_length):
            word = embedded_sentence[:, i, :]  # (batch_size, embedding_size)
            pre_h = self.lstm(pre_h, word)  # Only update h (hidden state)

        # print(pre_h.shape)
        # print(batch_size)
        # print( self.units)
        # print(self.input_length)
        # print(vocab_size)
        out = pre_h.view(batch_size * self.input_length, -1)
        # print("Out shape", out.shape)
        out =self.classification_model(out)

        return out # Pass the final hidden state into the classification network


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
        words = ['<sos>'] + line.split() + ['<eos>']
        tokens += len(words)
        for word in words:
          self.dictionary.add_word(word)
          # Create a 1-D tensor which contains index of all the words in the file with the help of word2idx
    ids = torch.LongTensor(tokens)
    token = 0
    with open(path, 'r') as f:
      for line in f:
        words =  ['<sos>']  + line.split() + ['<eos>']
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


embed_size = 100    # Embedding layer size , input to the LSTM
hidden_size = 1024  # Hidden size of LSTM units
num_layers = 1      # no LSTMs stacked
num_epochs = 10    # total no of epochs
batch_size = 25     # batch size
seq_length = 20     # sequence length
learning_rate = 0.002 # learning rate

corpus = Corpus()

ids = corpus.get_data('data/compliments.txt',batch_size)
# What is the vocabulary size ?
vocab_size = len(corpus.dictionary)
print(vocab_size)
# model = LSTM(vocab_size, embed_size, hidden_size, num_layers)
model = MinRNN(units=100, embedding_size=embed_size, vocab_size=vocab_size, input_length=20)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i in range(0, ids.size(1) - seq_length, seq_length):

        # move with seq length from the the starting index and move till - (ids.size(1) - seq_length)

        # prepare mini-batch inputs and targets
        inputs = ids[:, i:i + seq_length]  # fetch words for one seq length
        targets = ids[:, (i + 1):(i + 1) + seq_length]  # shifted by one word from inputs

        outputs = model(inputs)
        loss = criterion(outputs, targets.reshape(-1))
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

    #     total_loss += loss.item()
    #
    #     prediction = torch.sigmoid(outputs.squeeze())
    #     prediction = (prediction >= 0.5).float()
    #     correct += (prediction == labels).sum().item()
    #     total += labels.size(0)
    #
    # avg_loss = total_loss / len(train_loader)
    # avg_acc = correct / total
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))




# Test the model
with torch.no_grad():
    with open('results.txt', 'w') as f:
        # intial hidden ane cell states
        # state = (torch.zeros(num_layers, 1, hidden_size),
        #          torch.zeros(num_layers, 1, hidden_size))

        # Select one word id randomly and convert it to shape (1,1)
        # input = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1)
        # (min , max , shape) , convert to long tensor and make it a shape of 1,1
        # print(input.shape)

        input = torch.randint(0, 1, (1,)).long().unsqueeze(1)
        logVal = 0
        # print(input.shape)
        for i in range(10):
            output = model(input)

            # Sample a word id from the exponential of the output
            prob = output.exp()
            prob = output
            # prob = prob[0,:]
            for i in range(20):
                prob1 = prob[i, :]

                word_id = torch.multinomial(prob1, num_samples=1).item()
                prob1 = prob1 / torch.sum(prob1)
                logVal += torch.log(prob1[word_id])
                input.fill_(word_id)
                word = corpus.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

            # word_id = torch.multinomial(prob, num_samples=1).item()

            # print(word_id)

            # Replace the input with sampled word id for the next time step
            # input.fill_(word_id)

            # Write the results to file
            # word = corpus.dictionary.idx2word[word_id]
            # word = '\n' if word == '<eos>' else word + ' '
            # f.write(word)
            logVal = -1 / 20 * (logVal)
            perplexity = torch.pow(2, logVal)
            print("perplexity", perplexity)

            if (i + 1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i + 1, 500, 'results.txt'))
