import numpy as np
from random import uniform
import time




class LSTM:
    def __init__(self, char_to_idx, idx_to_char, vocab_size, n_h=100, seq_len=25,
                 epochs=10, lr=0.001, beta1=0.9, beta2=0.999, batch_size=25):
        """
        Implementation of simple character-level LSTM using Numpy
        """
        self.char_to_idx = char_to_idx  # characters to indices mapping
        self.idx_to_char = idx_to_char  # indices to characters mapping
        self.vocab_size = vocab_size  # no. of unique characters in the training data
        self.n_h = n_h  # no. of units in the hidden layer
        self.seq_len = seq_len  # no. of time steps, also size of mini batch
        self.epochs = epochs  # no. of training iterations
        self.lr = lr  # learning rate
        self.beta1 = beta1  # 1st momentum parameter
        self.beta2 = beta2  # 2nd momentum parameter
        self.params_count = 0
        # -----initialise weights and biases-----#
        self.params = {}
        std = (1.0 / np.sqrt(self.vocab_size + self.n_h))  # Xavier initialisation

        # forget gate
        self.params["Wfh"] = np.random.randn(self.vocab_size, self.n_h)
        self.params["Wf"] = np.random.randn(self.n_h, self.n_h) * std
        self.params_count += self.n_h * (self.n_h + self.vocab_size)
        self.params["bf"] = np.ones((self.n_h, 1))
        self.params_count += self.n_h
        # input gate
        self.params["Wxh"] = np.random.randn(self.vocab_size, self.n_h)
        self.params["Wi"] = np.random.randn(self.n_h, self.n_h) * std
        self.params_count += self.n_h * (self.n_h + self.vocab_size)
        self.params["bi"] = np.zeros((self.n_h, 1))

        # cell gate
        self.params["Wch"] = np.random.randn(self.vocab_size, self.n_h)

        self.params["Wc"] = np.random.randn(self.n_h, self.n_h) * std
        self.params_count += self.n_h * (self.n_h + self.vocab_size)
        self.params["bc"] = np.zeros((self.n_h, 1))
        self.params_count += self.n_h

        # output gate
        self.params["Woh"] = np.random.randn(self.vocab_size, self.n_h)

        self.params["Wo"] = np.random.randn(self.n_h, self.n_h) * std
        self.params_count += self.n_h * (self.n_h + self.vocab_size)
        self.params["bo"] = np.zeros((self.n_h, 1))
        self.params_count += self.n_h

        # output
        self.params["Wv"] = np.random.randn(self.vocab_size, self.n_h) * \
                            (1.0 / np.sqrt(self.vocab_size))
        self.params_count += self.n_h * self.vocab_size
        self.params["bv"] = np.zeros((self.vocab_size, 1))
        self.params_count += self.vocab_size

        # -----initialise gradients and Adam parameters-----#
        self.grads = {}
        self.adam_params = {}

        for key in self.params:
            self.grads["d" + key] = np.zeros_like(self.params[key])
            self.adam_params["m" + key] = np.zeros_like(self.params[key])
            self.adam_params["v" + key] = np.zeros_like(self.params[key])

        self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_len
        return

    def get_param_count(self):
        return self.params_count

    def sigmoid(self, x):
        """
        Smoothes out values in the range of [0,1]
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        Normalizes output into a probability distribution
        """
        e_x = np.exp(x - np.max(x))  # max(x) subtracted for numerical stability
        return e_x / np.sum(e_x)

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
        Updates parameters with Adam
        """
        for key in self.params:
            self.adam_params["m" + key] += self.grads["d" + key] * self.grads["d" + key]
            self.params[key] -= self.grads["d" + key] * self.lr / (np.sqrt(self.adam_params["m" + key]) + 1e-8)


    def sample_pass(self, h_prev, c_prev, sample_size):
        x = np.zeros((self.vocab_size, 1))
        h = h_prev
        c = c_prev
        index_arr = []
        logVal = 0
        for t in range(sample_size):
            y_hat, _, h, _, c, _, _, _, _ = self.forward_step(x, h, c)

            # get a random index within the probability distribution of y_hat(ravel())
            idx = np.random.choice(range(self.vocab_size), p=y_hat.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            logVal += np.log(y_hat[idx])
            # find the char with the sampled index and concat to the output string
            index_arr.append(idx)

        return index_arr, logVal

    def sample(self, h_prev, c_prev, sample_size):
        """
        Outputs a sample sequence from the model
        """
        x = np.zeros((self.vocab_size, 1))
        h = h_prev
        c = c_prev
        sample_string = ""
        logVal =0
        for t in range(sample_size):
            y_hat, _, h, _, c, _, _, _, _ = self.forward_step(x, h, c)

            # get a random index within the probability distribution of y_hat(ravel())
            idx = np.random.choice(range(self.vocab_size), p=y_hat.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            logVal += np.log(y_hat[idx])
            # find the char with the sampled index and concat to the output string
            char = self.idx_to_char[idx]
            sample_string += char
        logVal = -1 / sample_size * (logVal)
        perplexity = np.power(2, logVal)
        print("perplexity", perplexity)
        return sample_string

    def forward_step(self, x, h_prev, c_prev):
        """
        Implements the forward propagation for one time step
        """
        # z = np.row_stack((h_prev))
        z = h_prev

        f = self.sigmoid(np.dot(self.params["Wxh"].T,x) + np.dot(self.params["Wf"], z) + self.params["bf"])
        i = self.sigmoid(np.dot( self.params["Wfh"].T,x) + np.dot(self.params["Wi"], z) + self.params["bi"])
        o = self.sigmoid(np.dot( self.params["Woh"].T,x) + np.dot(self.params["Wo"], z) + self.params["bo"])

        c_bar = np.tanh( np.dot(self.params["Wch"].T,x) + np.dot(self.params["Wc"], z) + self.params["bc"])
        c = f * c_prev + i * c_bar
        h = o * np.tanh(c)

        v = np.dot(self.params["Wv"], h) + self.params["bv"]
        y_hat = self.softmax(v)
        return y_hat, v, h, o, c, c_bar, i, f, z

    def backward_step(self, y, y_hat, dh_next, dc_next, c_prev, z, x, f, i, c_bar, c, o, h):
        """
        Implements the backward propagation for one time step
        """
        dv = np.copy(y_hat)
        dv[y] -= 1  # yhat - y
        # Update Output
        self.grads["dWv"] += np.dot(dv, h.T)
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

            y_hat[t], v[t], h[t], o[t], c[t], c_bar[t], i[t], f[t], z[t] = \
                self.forward_step(x[t], h[t - 1], c[t - 1])

            loss += -np.log(y_hat[t][y_batch[t], 0])

        self.reset_grads()

        dh_next = np.zeros_like(h[0])
        dc_next = np.zeros_like(c[0])

        for t in reversed(range(self.seq_len)):
            dh_next, dc_next = self.backward_step(y_batch[t], y_hat[t], dh_next,
                                                  dc_next, c[t - 1], z[t], x[t], f[t], i[t],
                                                  c_bar[t], c[t], o[t], h[t])
        return loss, h[self.seq_len - 1], c[self.seq_len - 1]

    def gradient_check(self, x, y, h_prev, c_prev, num_checks=10, delta=1e-6):
        """
        Checks the magnitude of gradients against expected approximate values
        """
        print("**********************************")
        print("Gradient check...\n")

        _, _, _ = self.forward_backward(x, y, h_prev, c_prev)
        grads_numerical = self.grads

        for key in self.params:
            print("---------", key, "---------")
            test = True

            dims = self.params[key].shape
            grad_numerical = 0
            grad_analytical = 0

            for _ in range(num_checks):  # sample 10 neurons

                idx = int(uniform(0, self.params[key].size))
                old_val = self.params[key].flat[idx]

                self.params[key].flat[idx] = old_val + delta
                J_plus, _, _ = self.forward_backward(x, y, h_prev, c_prev)

                self.params[key].flat[idx] = old_val - delta
                J_minus, _, _ = self.forward_backward(x, y, h_prev, c_prev)

                self.params[key].flat[idx] = old_val

                grad_numerical += (J_plus - J_minus) / (2 * delta)
                grad_analytical += grads_numerical["d" + key].flat[idx]

            grad_numerical /= num_checks
            grad_analytical /= num_checks

            rel_error = abs(grad_analytical - grad_numerical) / abs(grad_analytical + grad_numerical)

            # if rel_error > 1e-2:
            #     if not (grad_analytical < 1e-6 and grad_numerical < 1e-6):
            #         test = False
            #         assert (test)

            print('Approximate: \t%e, Exact: \t%e =>  Error: \t%e' % (grad_numerical, grad_analytical, rel_error))
        print("\nTest successful!")
        print("**********************************\n")
        return

    def train(self, X, verbose=True):
        """
        Main method of the LSTM class where training takes place
        """
        J = []  # to store losses

        num_batches = len(X) // self.seq_len
        X_trimmed = X[: num_batches * self.seq_len]  # trim input to have full sequences

        for epoch in range(self.epochs):
            epoch_start = time.time()

            h_prev = np.zeros((self.n_h, 1))
            c_prev = np.zeros((self.n_h, 1))
            batch_time = time.time()
            for j in range(0, len(X_trimmed) - self.seq_len, self.seq_len):
                    # prepare batches

                    x_batch = [self.char_to_idx[ch] for ch in X_trimmed[j: j + self.seq_len]]
                    y_batch = [self.char_to_idx[ch] for ch in X_trimmed[j + 1: j + self.seq_len + 1]]

                    loss, h_prev, c_prev = self.forward_backward(x_batch, y_batch, h_prev, c_prev)

                    # smooth out loss and store in list
                    self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
                    J.append(self.smooth_loss)

                    # check gradients
                    if epoch == 0 and j == 0:
                        self.gradient_check(x_batch, y_batch, h_prev, c_prev, num_checks=10, delta=1e-7)

                    self.clip_grads()

                    batch_num = epoch * self.epochs + j / self.seq_len + 1
                    self.update_params(batch_num)
                    # self.update_params_ada(batch_num)

                    # print out loss and sample string
                    if j % 80000 == 0:
                        # print("Batch time",time.time()- batch_time)
                        s = self.sample(h_prev, c_prev, sample_size=200)
                        print(s, "\n")
                        batch_time = time.time()
            epoch_end = time.time()- epoch_start
            print('Epoch:', epoch, 'time', epoch_end, '\tBatch:', j, "-", j + self.seq_len,
                  '\tLoss:', round(self.smooth_loss, 2))
            s = self.sample(h_prev, c_prev, sample_size=200)
            print(s, "\n")

        return J, self.params