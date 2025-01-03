
import torch


class Tokenize():
    def __init__(self):
        self.tokens = None
        self.wordToId = {}
        self.IdToWord = {}
        self.idx = 0
        self.id_tensor = None

    def create_id_for_word(self, word):
        if word not in self.wordToId:
            self.IdToWord[self.idx] = word
            self.wordToId[word] = self.idx
            self.idx += 1


    def tokenize_text(self, path):
        with open(path, 'r') as f:
            token = 0
            for line in f:
                words = line.split()  + ['<eos']
                token += len(words)
                for word in words:
                    self.create_id_for_word(word)

            self.id_tensor = torch.LongTensor(token)

            with open(path, 'r') as f:
                token = 0
                for line in f:
                    words = line.split() + ['<eos']
                    for word in words:
                        self.id_tensor[token] = self.wordToId[word]
                        token += 1



    def create_batches(self, batch_size=20):
        num_batches = self.id_tensor.shape[0] // batch_size
        self.id_tensor = self.id_tensor[: num_batches * batch_size]
        self.id_tensor = self.id_tensor.view(batch_size , -1)
        return self.id_tensor

    def corpus_create(self, path):
        self.tokenize_text(path)
        return self.create_batches(20)
















