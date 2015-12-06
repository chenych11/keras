from __future__ import absolute_import
import theano
import theano.tensor as T

from .. import activations, initializations, regularizers, constraints
from ..layers.core import Layer, MaskedLayer
from ..utils.theano_utils import sharedX

from ..constraints import unitnorm
import logging
logger = logging.getLogger('keras.layers.embedding')

class Embedding(Layer):
    """
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    """
    def __init__(self, input_dim, output_dim, init='uniform',
                 W_regularizer=None, activity_regularizer=None, W_constraint=None,
                 mask_zero=False, weights=None):

        # super(Embedding, self).__init__()
        Layer.__init__(self)
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.imatrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.mask_zero = mask_zero

        self.params = [self.W]

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.regularizers = []

        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if weights is not None:
            self.set_weights(weights)

        self.__output_slots = None

    def get_output_mask(self, train=False):
        if not self.mask_zero:
            return None
        else:
            X = self.get_input(train)
            return (T.ones_like(X) * (1 - T.eq(X, 0))).astype(theano.config.floatX)

    def get_output(self, train=False):
        if self.__output_slots is None:
            train_in = self.get_input(True)
            test_in = self.get_input(False)
            self.__output_slots = {True: self.W[train_in], False: self.W[test_in]}
        out = self.__output_slots[train]
        return out

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None}

from keras.optimizers import SubTensorInfo


class LookupTable(Layer):
    """
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    """
    def __init__(self, input_dim, output_dim, init='uniform',
                 W_regularizer=None, activity_regularizer=None, W_constraint=None,
                 mask_zero=False, weights=None):

        # super(Embedding, self).__init__()
        Layer.__init__(self)
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.imatrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.W.name = 'embeddings'
        self.mask_zero = mask_zero

        self.sub_embedding_param = None
        self.output_slot = {True: None, False: None}
        self.params = []

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if weights is not None:
            self.set_weights(weights)

    def get_input(self, train=False):
        if hasattr(self, 'previous'):
            # logger.warn('The LookupTable layer is getting input from previous layer in %s mode. '
            #             'This may be an issue if train and test mode returns different symbolic tensors.'
            #             'This layer is specifically designed to be the first layer of the network, '
            #             'though it works if the network\'s input is irrelevant to the train or test mode.' %
            #             'train' if train else 'test')
            return self.previous.get_output(train=train)
        else:
            return self.input

    def get_output_mask(self, train=False):
        if not self.mask_zero:
            return None
        else:
            X = self.get_input(train)
            return (T.ones_like(X) * (1 - T.eq(X, 0))).astype(theano.config.floatX)

    def on_connection_end(self):
        train_input = self.get_input(train=True)
        train_input_flat = train_input.flatten()
        test_input = self.get_input(train=False)

        uni_idx, inverse_idx = theano.tensor.extra_ops.Unique(False, True, False)(train_input_flat)

        self.sub_embedding_param = self.W[uni_idx]
        self.sub_embedding_param.subtensor_info = SubTensorInfo(self.sub_embedding_param, self.W,
                                                                uni_idx,
                                                                shape=(self.input_dim, self.output_dim))
        tmp = self.sub_embedding_param[inverse_idx]
        self.output_slot[True] = tmp.reshape(T.concatenate((train_input.shape, [self.output_dim])),
                                             ndim=train_input.ndim+1)
        self.output_slot[False] = self.W[test_input]
        self.params = [self.sub_embedding_param]

    def get_output(self, train=False):
        return self.output_slot[True] if train else self.output_slot[False]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None}


class WordContextProduct(Layer):
    """
        This layer turns a pair of words (a pivot word + a context word,
        ie. a word from the same context, or a random, out-of-context word),
        indentified by their index in a vocabulary, into two dense reprensentations
        (word representation and context representation).

        Then it returns activation(dot(pivot_embedding, context_embedding)),
        which can be trained to encode the probability
        of finding the context word in the context of the pivot word
        (or reciprocally depending on your training procedure).

        The layer ingests integer tensors of shape:
        (nb_samples, 2)
        and outputs a float tensor of shape
        (nb_samples, 1)

        The 2nd dimension encodes (pivot, context).
        input_dim is the size of the vocabulary.

        For more context, see Mikolov et al.:
            Efficient Estimation of Word reprensentations in Vector Space
            http://arxiv.org/pdf/1301.3781v3.pdf
    """
    def __init__(self, input_dim, proj_dim=128,
                 init='uniform', activation='sigmoid', weights=None):

        super(WordContextProduct, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.input = T.imatrix()
        # two different embeddings for pivot word and its context
        # because p(w|c) != p(c|w)
        self.W_w = self.init((input_dim, proj_dim))
        self.W_c = self.init((input_dim, proj_dim))

        self.params = [self.W_w, self.W_c]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train=False):
        X = self.get_input(train)
        w = self.W_w[X[:, 0]]  # nb_samples, proj_dim
        c = self.W_c[X[:, 1]]  # nb_samples, proj_dim

        dot = T.sum(w * c, axis=1)
        dot = theano.tensor.reshape(dot, (X.shape[0], 1))
        return self.activation(dot)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "proj_dim": self.proj_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__}
