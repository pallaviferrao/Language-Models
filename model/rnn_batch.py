#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class VanillaRNN:

    def __init__(self, char_to_idx, idx_to_char, vocab_size, hidden_layer_size=75,
                 seq_len=100, clip_rate=5, epochs=50, learning_rate=1e-2):
        """
        Implementation of simple character-level RNN using Numpy
        Major inspiration from karpathy/min-char-rnn.py
        """

        # assign instance variables
        self.char_to_idx = char_to_idx  # dictionary that maps characters in the vocabulary to an index
        self.idx_to_char = idx_to_char  # dictionary that maps indices to unique characters in the vocabulary

        self.vocab_size = vocab_size  # number of unique characters in the training data
        self.n_h = hidden_layer_size  # desirable number of units in the hidden layer

        self.seq_len = seq_len  # number of characters that will be fed to the RNN in each batch (also number of time steps)
        self.clip_rate = clip_rate  # maximum absolute value for the gradients, which are limited to avoid exploding gradients
        self.epochs = epochs  # number of training iterations
        self.learning_rate = learning_rate

        self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_len  # smoothing out loss as batch SGD is noisy

        # initialize parameters
        self.params = {}

        self.params["W_xh"] = torch.rand(self.vocab_size, self.n_h) * 0.01

        self.params["W_hh"] = torch.rand(self.n_h, self.n_h) * 0.01
        self.params["b_h"] = torch.zeros((1, self.n_h))

        self.params["W_hy"] = torch.rand(self.n_h, self.vocab_size) * 0.01
        self.params["b_y"] = torch.zeros((1, self.vocab_size))

        self.h0 = torch.zeros((1, self.n_h))  # value of hidden state at time step t = -1. This is updated over time

        # initialize gradients and memory parameters for Adagrad
        self.grads = {}
        self.m_params = {}

        self.beta1 = 0.9  # 1st momentum parameter
        self.beta2 = 0.999  # 2nd momentum parameter

        for key in self.params:
            self.grads["d" + key] = torch.zeros_like(self.params[key])
            self.m_params["m" + key] = torch.zeros_like(self.params[key])
            self.m_params["v" + key] = np.zeros_like(self.params[key])




    def softmax(self, z):
        exp_z = torch.exp(z - torch.max(z))
        return exp_z / exp_z.sum(dim=-1, keepdim=True)

    def forward_pass(self, X):
        """
        Implements the forward propagation for a RNN
        Input:
            X - input batch, one-hot-encoded
        Output:
            y_pred - output softmax probabilities
            h - hidden states
        """
        h = {}  # stores hidden states
        h[-1] = self.h0  # set initial hidden state at t=-1

        y_pred = {}  # stores softmax output probabilities

        # iterate over each character in the input sequence
        for t in range(self.seq_len):
            # print(len(X))

            X[t]= torch.from_numpy(np.float32(X[t]))
            # print(X[t].dtype)
            # print(self.params["W_xh"].dtype)

            h[t] = torch.tanh(
                torch.matmul(X[t], self.params["W_xh"]) + torch.matmul(h[t - 1], self.params["W_hh"]) + self.params["b_h"])


            y_pred[t] = self.softmax(torch.matmul(h[t], self.params["W_hy"]) + self.params["b_y"])


        self.ho = h[t]
        return y_pred, h


    def backward_pass(self, X, y, y_pred, h):
        """
        Implements the backward propagation for a RNN
        Input:
            X - input batch, one-hot-encoded
            y - label batch, one-hot-encoded
            y_pred - output softmax probabilities from forward propagation
            h - hidden states from forward propagation
        Output:
            grads - derivatives of every learnable parameter in the RNN
        """
        dh_next = torch.zeros_like(h[0])

        for t in reversed(range(self.seq_len)):
            dy = y_pred[t]
            # print(dy.shape)
            # print("y len:=", len(y))
            dy[0][torch.argmax(torch.from_numpy(y[t]))] -= 1  # predicted y - actual y
            # print("dy2", dy.shape)
            # dy = y[t] - y_pred[t]
            # dy =  dy[0][np.argmax(y_pred[t])] - y[t]
            # dy = (torch.argmax(y_pred[t]) - y[t]).to(torch.float32)
            # dy = y_pred[t][0][np.argmax(y[t])] -1
            self.grads["dW_hy"] += np.dot(h[t].T, dy)
            self.grads["db_y"] += dy

            dhidden = (1 - h[t] ** 2) * (torch.matmul(dy, self.params["W_hy"].T) + dh_next)
            dh_next = torch.matmul(dhidden, self.params["W_hh"].T)

            self.grads["dW_hh"] += torch.matmul(h[t - 1].T, dhidden)
            self.grads["dW_xh"] += torch.matmul(X[t].T, dhidden)
            self.grads["db_h"] += dhidden

            # clip gradients to mitigate exploding gradients
        for grad, key in enumerate(self.grads):
            torch.clip(self.grads[key], -self.clip_rate, self.clip_rate, out=self.grads[key])
        return


    def update_params(self):
        """
        Updates parameters with Adagrad
        """
        for key in self.params:
            self.m_params["m" + key] += self.grads["d" + key] * self.grads["d" + key]
            self.params[key] -= self.grads["d" + key] * self.learning_rate / (np.sqrt(self.m_params["m" + key]) + 1e-8)

    def update_params_adam(self,batch_num):
        """
        Updates parameters with Adagrad
        """
        for key in self.params:
            self.m_params["m" + key] = self.m_params["m" + key] * self.beta1 + \
                                          (1 - self.beta1) * self.grads["d" + key]
            self.m_params["v" + key] = self.m_params["v" + key] * self.beta2 + \
                                          (1 - self.beta2) * self.grads["d" + key] ** 2

            m_correlated = self.m_params["m" + key] / (1 - self.beta1 ** batch_num)
            v_correlated = self.m_params["v" + key] / (1 - self.beta2 ** batch_num)
            self.params[key] -= self.lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)
        return

    def sample_pass(self,sample_size, start_index ):
        s = []

        x = torch.zeros((1, self.vocab_size))
        x[0][start_index] = 1
        logVal= 0
        for i in range(sample_size):
            # forward propagation
            h = torch.tanh(torch.matmul(x, self.params["W_xh"]) + torch.matmul(self.h0, self.params["W_hh"]) + self.params["b_h"])
            y_pred = self.softmax(torch.matmul(h, self.params["W_hy"]) + self.params["b_y"])

            # index = np.random.choice(range(self.vocab_size), p=y_pred.ravel())
            index = torch.multinomial(y_pred, num_samples=1)  # (B, 1)
            logVal += torch.log(y_pred[0][index])
            # index = np.argmax(y_pred)
            # set x-one_hot_vector for the next character
            x = torch.zeros((1, self.vocab_size))
            x[0][index] = 1
            s.append(index)
        return s, logVal
