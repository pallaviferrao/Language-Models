import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from model.lstm_model_nn import LSTM
from Tokenize.tokenize import Corpus


#Word LSTM from torch
embed_size = 128    # Embedding layer size , input to the LSTM
hidden_size = 1024  # Hidden size of LSTM units
num_layers = 2      # no LSTMs stacked
num_epochs = 10    # total no of epochs
batch_size = 20     # batch size
seq_length = 20     # sequence length
learning_rate = 0.002 # learning rate
corpus = Corpus()
ids = corpus.get_data('./data/compliments.txt',batch_size)

print("id",corpus.get_id('<sos>'))
print("test1", torch.randint(0, 1, (1,)).long().unsqueeze(1))
# ids tensors contain all the index of each words
print(ids.shape)


# What is the vocabulary size ?
vocab_size = len(corpus.dictionary)
print(vocab_size)

num_batches = ids.shape[1] // seq_length
print(num_batches)
model = LSTM( vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# to Detach the Hidden and Cell states from previous history
def detach(states):
    return [state.detach() for state in states]


for epoch in range(num_epochs):
    # initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size),
              torch.zeros(num_layers, batch_size, hidden_size))

    for i in range(0, ids.size(1) - seq_length, seq_length):

        # move with seq length from the the starting index and move till - (ids.size(1) - seq_length)

        # prepare mini-batch inputs and targets
        inputs = ids[:, i:i + seq_length]  # fetch words for one seq length
        targets = ids[:, (i + 1):(i + 1) + seq_length]  # shifted by one word from inputs


        states = detach(states)

        outputs, states = model(inputs, states)
        # loss = criterion(outputs, targets.reshape(-1))
        loss = criterion(outputs, targets.reshape(-1))
        model.zero_grad()
        loss.backward()

        # The gradients are clipped in the range [-clip_value, clip_value]. This is to prevent the exploding gradient problem
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // seq_length
        if step % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Test the model
with torch.no_grad():
    with open('results.txt', 'w') as f:
        # intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size),
                 torch.zeros(num_layers, 1, hidden_size))

        # Select one word id randomly and convert it to shape (1,1)
        # input = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1)
        # (min , max , shape) , convert to long tensor and make it a shape of 1,1
        for sample in range(5):
            input = torch.randint(0, 1, (1,)).long().unsqueeze(1)
            logVal = 0
            for i in range(10):
                output, _ = model(input, state)

                # Sample a word id from the exponential of the output
                prob = output.exp()

                word_id = torch.multinomial(prob, num_samples=1).item()
                # print(word_id)
                prob = prob/torch.sum(prob)

                logVal += torch.log(prob[:,word_id])
                # print(logVal)
                # Replace the input with sampled word id for the next time step
                input.fill_(word_id)

                # Write the results to file
                word = corpus.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

                if (i + 1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i + 1, 500, 'results.txt'))

            logVal = -1/10*(logVal)
            perplexity = torch.pow(2, logVal)
            print("perplexity", perplexity)