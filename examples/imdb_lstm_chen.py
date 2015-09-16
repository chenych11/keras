#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Train a LSTM on the IMDB sentiment classification task.

    The dataset is actually too small for LSTM to be of any advantage
    compared to simpler, much faster methods such as TF-IDF+LogReg.

    Notes:

    - RNNs are tricky. Choice of batch size is important,
    choice of loss and optimizer is critical, etc.
    Some configurations won't converge.

    - LSTM loss decrease patterns during training can be quite different
    from what you see with CNNs/MLPs/etc.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm_chen.py
"""
from __future__ import absolute_import
from __future__ import print_function
__author__ = 'Yunchuan Chen'
from theano import tensor as T
import theano
import numpy as np
from keras.preprocessing import sequence
# from keras.optimizers import SGD, RMSprop, Adagrad
# from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import MaskedLayer, Recurrent
from keras import activations, initializations
from keras.utils.theano_utils import shared_zeros
from keras.datasets import imdb


class LSTMLayer(Recurrent):
    """
         Self-defined LSTM layer
        optimized version: Not using mask in _step function and tensorized computation.
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    """
    def __init__(self, input_dim, output_dim=128, train_init_cell=True, train_init_h=True,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 input_activation='tanh', gate_activation='hard_sigmoid', output_activation='tanh',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(LSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)
        self.input = T.tensor3()
        self.time_range = None

        W_z = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_z = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_z = shared_zeros(self.output_dim)

        W_i = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_i = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_i = shared_zeros(self.output_dim)

        W_f = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_f = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_f = self.forget_bias_init(self.output_dim)

        W_o = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_o = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_o = shared_zeros(self.output_dim)

        self.h_m1 = shared_zeros(shape=(1, self.output_dim), name='h0')
        self.c_m1 = shared_zeros(shape=(1, self.output_dim), name='c0')

        W = np.vstack((W_z[np.newaxis, :, :],
                       W_i[np.newaxis, :, :],
                       W_f[np.newaxis, :, :],
                       W_o[np.newaxis, :, :]))  # shape = (4, input_dim, output_dim)
        R = np.vstack((R_z[np.newaxis, :, :],
                       R_i[np.newaxis, :, :],
                       R_f[np.newaxis, :, :],
                       R_o[np.newaxis, :, :]))  # shape = (4, output_dim, output_dim)
        self.W = theano.shared(W, name='Input to hidden weights (zifo)', borrow=True)
        self.R = theano.shared(R, name='Recurrent weights (zifo)', borrow=True)
        self.b = theano.shared(np.zeros(shape=(4, self.output_dim), dtype=theano.config.floatX),
                               name='bias', borrow=True)

        self.params = [self.W, self.R]
        if train_init_cell:
            self.params.append(self.c_m1)
        if train_init_h:
            self.params.append(self.h_m1)

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              Y_t,      # sequence
              h_tm1, c_tm1,   # output_info
              R):          # non_sequence
        # h_mask_tm1 = mask_tm1 * h_tm1
        # c_mask_tm1 = mask_tm1 * c_tm1
        G_tm1 = T.dot(h_tm1, R)
        M_t = Y_t + G_tm1
        z_t = self.input_activation(M_t[:, 0, :])
        ifo_t = self.gate_activation(M_t[:, 1:, :])
        i_t = ifo_t[:, 0, :]
        f_t = ifo_t[:, 1, :]
        o_t = ifo_t[:, 2, :]
        # c_t_cndt = f_t * c_tm1 + i_t * z_t
        # h_t_cndt = o_t * self.output_activation(c_t_cndt)
        c_t = f_t * c_tm1 + i_t * z_t
        h_t = o_t * self.output_activation(c_t)
        # h_t = mask * h_t_cndt + (1-mask) * h_tm1
        # c_t = mask * c_t_cndt + (1-mask) * c_tm1
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        # mask = self.get_padded_shuffled_mask(train, X, pad=0)
        mask = self.get_input_mask(train=train)
        ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32')
        max_time = T.max(ind)
        X = X.dimshuffle((1, 0, 2))
        Y = T.dot(X, self.W) + self.b
        # h0 = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        h0 = T.repeat(self.h_m1, X.shape[1], axis=0)
        c0 = T.repeat(self.c_m1, X.shape[1], axis=0)

        [outputs, _], updates = theano.scan(
            self._step,
            sequences=Y,
            outputs_info=[h0, c0],
            non_sequences=[self.R], n_steps=max_time,
            truncate_gradient=self.truncate_gradient, strict=True,
            allow_gc=theano.config.scan.allow_gc)

        res = T.concatenate([h0.dimshuffle('x', 0, 1), outputs], axis=0).dimshuffle((1, 0, 2))
        if self.return_sequences:
            return res
        #return outputs[-1]
        return res[T.arange(mask.shape[0], dtype='int32'), ind]

    def set_init_cell_parameter(self, is_param=True):
        if is_param:
            if self.c_m1 not in self.params:
                self.params.append(self.c_m1)
        else:
            self.params.remove(self.c_m1)

    def set_init_h_parameter(self, is_param=True):
        if is_param:
            if self.h_m1 not in self.params:
                self.params.append(self.h_m1)
        else:
            self.params.remove(self.h_m1)

    # def get_time_range(self, train):
    #     mask = self.get_input_mask(train=train)
    #     ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32')
    #     self.time_range = ind
    #     return ind


class MeanPooling(MaskedLayer):
    """
        Self-defined pooling layer
        Global Mean Pooling Layer
    """
    def __init__(self, start=1):
        super(MeanPooling, self).__init__()
        self.start = start

    # def supports_masked_input(self):
    #     return False
    def get_output_mask(self, train=False):
        return None

    def get_output(self, train=False):
        data = self.get_input(train=train)
        mask = self.get_input_mask(train=train)
        mask = mask.dimshuffle((0, 1, 'x'))
        return (data[:, self.start:] * mask).mean(axis=1)

    def get_config(self):
        return {"name": self.__class__.__name__}


np.random.seed(1337)  # for reproducibility
max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model (The output of the last timestep is applied for classification)...')
model = Sequential()
model.add(Embedding(max_features, 128, mask_zero=True))
model.add(LSTMLayer(128, 128))
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)

print('=' * 60)

print('Build model (The outputs of all timesteps are applied for classification)...')
model = Sequential()
model.add(Embedding(max_features, 128, mask_zero=True))
model.add(LSTMLayer(128, 128, return_sequences=True))
model.add(MeanPooling())
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)