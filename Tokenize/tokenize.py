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

    def get_id(self, word):
        return self.dictionary.word2idx[word]

    def get_word_for_id(self, id):
        return self.dictionary.idx2word[id]

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
                words = ['<sos>'] + line.split() + ['<eos>']
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

    def tokenize(self, path):
        """Tokenizes a text file."""
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


    def get_data_no_batch(self, path, batch_size=20):
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
                words = ['<sos>'] + line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        # no of required batches
        num_batches = ids.shape[0] // batch_size
        # Remove the remainder from the last batch , so that always batch size is constant
        ids = ids[:num_batches * batch_size]
        # return (batch_size,num_batches)
        return ids