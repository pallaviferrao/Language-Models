import numpy as np
from random import uniform
import time
import model.utils
from model.utils import sigmoid
from model.utils import softmax
class LSTM:
    def __init__(self, char_to_idx, idx_to_char, vocab_size, n_h=100, seq_len=25,
                 epochs=10, lr=0.001, beta1=0.9, beta2=0.999, batch_size=25):
        """
        Implementation of simple character-level LSTM using Numpy
        """
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size
        self.n_h = n_h
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.params_count = 0
        self.params = {}
        std = (1.0 / np.sqrt(self.vocab_size + self.n_h))  # Xavier initialisation
        # forget gate
        self.params["Wfh"],  self.params["Wf"],self.params["bf"] = self.initialize_var(std)
        # input gate
        self.params["Wxh"],self.params["Wi"],self.params["bi"] = self.initialize_var(std)
        # cell gate
        self.params["Wch"],self.params["Wc"],self.params["bc"] =  self.initialize_var(std)
        # output gate
        self.params["Woh"],self.params["Wo"],self.params["bo"] =  self.initialize_var(std)

        # output
        self.params["Wv"] = np.random.randn(self.vocab_size, self.n_h) * \
                            (1.0 / np.sqrt(self.vocab_size))
        self.params_count += self.n_h * self.vocab_size
        self.params["bv"] = np.zeros((self.vocab_size, 1))
        self.params_count += self.vocab_size

        # -----initialise gradients and Adam parameters-----#
        self.grads = {}
        self.adam_params = {}
        self.initialize_grad()
        self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_len

    def initialize_grad(self):
        for key in self.params:
            self.grads["d" + key] = np.zeros_like(self.params[key])
            self.adam_params["m" + key] = np.zeros_like(self.params[key])
            self.adam_params["v" + key] = np.zeros_like(self.params[key])

    def initialize_var(self, std):
        input_param = np.random.randn(self.vocab_size, self.n_h)
        hidden_layer = np.random.randn(self.n_h, self.n_h) * std
        bias = np.ones((self.n_h, 1))
        self.params_count += self.n_h * (self.n_h + self.vocab_size)
        self.params_count += self.n_h
        return input_param, hidden_layer, bias

    def get_param_count(self):
        return self.params_count

    def clip_grads(self):
        """
        Limits the magnitude of gradients to avoid exploding gradients
        """
        for key in self.grads:
            np.clip(self.grads[key], -5, 5, out=self.grads[key])
        return

    def reset_grads(self):
        """
        Resets gradients to zero before each backpropagation
        """
        for key in self.grads:
            self.grads[key].fill(0)
        return

    def update_params(self, batch_num):
        """
        Updates parameters with Adam
        """
        for key in self.params:
            self.adam_params["m" + key] = self.adam_params["m" + key] * self.beta1 + \
                                          (1 - self.beta1) * self.grads["d" + key]
            self.adam_params["v" + key] = self.adam_params["v" + key] * self.beta2 + \
                                          (1 - self.beta2) * self.grads["d" + key] ** 2

            m_correlated = self.adam_params["m" + key] / (1 - self.beta1 ** batch_num)
            v_correlated = self.adam_params["v" + key] / (1 - self.beta2 ** batch_num)
            self.params[key] -= self.lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)
        return

    def update_params_ada(self, batch_num):
        """
        Updates parameters with Ada
        """
        for key in self.params:
            self.adam_params["m" + key] += self.grads["d" + key] * self.grads["d" + key]
            self.params[key] -= self.grads["d" + key] * self.lr / (np.sqrt(self.adam_params["m" + key]) + 1e-8)

    def sample(self, h_prev, c_prev, sample_size):
        x = np.zeros((self.vocab_size, 1))
        h = h_prev
        c = c_prev
        index_arr = []
        logVal = 0
        for t in range(sample_size):
            y_hat,  h, _, c, _, _, _ = self.forward_step(x, h, c)

            # get a random index within the probability distribution of y_hat(ravel())
            idx = np.random.choice(range(self.vocab_size), p=y_hat.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            logVal += np.log(y_hat[idx])
            # find the char with the sampled index and concat to the output string
            index_arr.append(idx)
        return index_arr, logVal

    def forward_step(self, x, h_prev, c_prev):
        forget_gate = sigmoid(np.dot(self.params["Wxh"].T, x) + np.dot(self.params["Wf"], h_prev) + self.params["bf"])
        input_gate = sigmoid(np.dot(self.params["Wfh"].T, x) + np.dot(self.params["Wi"], h_prev) + self.params["bi"])
        output_gate = sigmoid(np.dot(self.params["Woh"].T, x) + np.dot(self.params["Wo"], h_prev) + self.params["bo"])

        c_bar = np.tanh(np.dot(self.params["Wch"].T, x) + np.dot(self.params["Wc"], h_prev) + self.params["bc"])
        c = forget_gate * c_prev + input_gate * c_bar
        hidden_state = output_gate * np.tanh(c)

        v = np.dot(self.params["Wv"], hidden_state) + self.params["bv"]
        y_hat = softmax(v)

        return y_hat, hidden_state, output_gate, c, c_bar, input_gate, forget_gate

    def backward_step(self, y, y_hat, dh_next, dc_next, c_prev, z, x, f, i, c_bar, c, o):
        """
        Implements the backward propagation for one time step
        """
        dv = np.copy(y_hat)
        dv[y] -= 1  # yhat - y
        # Update Output
        self.grads["dWv"] += np.dot(dv, z.T)
        self.grads["dbv"] += dv
        # Store Hidden state for H(t)
        dh = np.dot(self.params["Wv"].T, dv)
        dh += dh_next

        do = dh * np.tanh(c)
        da_o = do * o * (1 - o)
        self.grads["dWoh"] += np.dot(x,da_o.T)
        self.grads["dWo"] += np.dot(da_o, z.T)
        self.grads["dbo"] += da_o

        dc = dh * o * (1 - np.tanh(c) ** 2)
        dc += dc_next
        dc_bar = dc * i
        da_c = dc_bar * (1 - c_bar ** 2)
        self.grads["dWch"] += np.dot(x, da_c.T)
        self.grads["dWc"] += np.dot(da_c, z.T)
        self.grads["dbc"] += da_c

        di = dc * c_bar
        da_i = di * i * (1 - i)
        self.grads["dWxh"] += np.dot(x, da_i.T)
        self.grads["dWi"] += np.dot(da_i, z.T)
        self.grads["dbi"] += da_i

        df = dc * c_prev
        da_f = df * f * (1 - f)
        self.grads["dWfh"] += np.dot(x, da_f.T)
        self.grads["dWf"] += np.dot(da_f, z.T)
        self.grads["dbf"] += da_f

        dz = (np.dot(self.params["Wf"].T, da_f)
              + np.dot(self.params["Wi"].T, da_i)
              + np.dot(self.params["Wc"].T, da_c)
              + np.dot(self.params["Wo"].T, da_o))

        dh_prev = dz[:self.n_h, :]
        dc_prev = f * dc
        return dh_prev, dc_prev

    def forward_backward(self, x_batch, y_batch, h_prev, c_prev):
        """
        Implements the forward and backward propagation for one batch
        """
        x, z = {}, {}
        f, i, c_bar, c, o = {}, {}, {}, {}, {}
        y_hat, v, h = {}, {}, {}

        # Values at t= - 1
        h[-1] = h_prev
        c[-1] = c_prev

        loss = 0
        for t in range(self.seq_len):
            x[t] = np.zeros((self.vocab_size, 1))
            x[t][x_batch[t]] = 1
            y_hat[t], h[t], o[t], c[t], c_bar[t], i[t], f[t]= self.forward_step(x[t], h[t - 1], c[t - 1])
            loss += -np.log(y_hat[t][y_batch[t], 0])
        self.reset_grads()

        dh_next = np.zeros_like(h[0])
        dc_next = np.zeros_like(c[0])

        for t in reversed(range(self.seq_len)):
            dh_next, dc_next = self.backward_step(y_batch[t], y_hat[t], dh_next,
                                                  dc_next, c[t - 1], h[t], x[t], f[t], i[t],
                                                  c_bar[t], c[t], o[t])
        return loss, h[self.seq_len - 1], c[self.seq_len - 1]