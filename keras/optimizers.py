from __future__ import absolute_import
import theano
import theano.tensor as T

from .utils.theano_utils import shared_zeros, shared_scalar, floatX
from .utils.generic_utils import get_from_module
from six.moves import zip
import numpy as np

float_t = theano.config.floatX

def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g * c / n, g)
    return g


def kl_divergence(p, p_hat):
    return p_hat - p + p * T.log(p / p_hat)


class SubTensorInfo(object):
    def __init__(self, subset, base, index, shape=None):
        super(SubTensorInfo, self).__init__()
        # assert isinstance(base, Shared_t) and isinstance(index, T.TensorVariable) and \
        #        isinstance(subset, T.TensorVariable), 'base: %s, index: %s, subset: %s' % \
        #                                              (type(base), type(index), type(subset))
        self.variable = subset
        self.base = base
        self.idx = index
        if shape is not None:
            self.shape = shape
        else:
            self.shape = tuple(base.shape.eval())


class Optimizer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []

    def get_state(self):
        return [u[0].get_value() for u in self.updates]

    def set_state(self, value_list):
        assert len(self.updates) == len(value_list)
        for u, v in zip(self.updates, value_list):
            u[0].set_value(floatX(v))

    def get_updates(self, params, constraints, loss):
        raise NotImplementedError

    def get_gradients(self, loss, params):

        grads = T.grad(loss, params)

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]

        return grads

    def get_config(self):
        return {"name": self.__class__.__name__}


class SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, *args, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)
        self.lr = shared_scalar(lr)
        self.momentum = shared_scalar(momentum)
        self.decay = shared_scalar(decay)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, c in zip(params, grads, constraints):
            try:
                m = shared_zeros(p.get_value().shape)  # momentum
                v = self.momentum * m - lr * g  # velocity
                self.updates.append((m, v))

                if self.nesterov:
                    new_p = p + self.momentum * v - lr * g
                else:
                    new_p = p + v
                self.updates.append((p, c(new_p)))  # apply constraints
            except AttributeError:
                # subtensor update:
                # v_t = (self.momentum * p.sub_momenton) - lr * g
                # if self.nesterov:
                #     new_p = T.inc_subtensor(p, self.momentum * v_t - lr * g)
                # else:
                #     new_p = T.inc_subtensor(p, v_t)
                # self.updates.append((p.momenton_whole, T.set_subtensor(p.sub_momenton, v_t)))
                new_p = T.inc_subtensor(p, -lr * g)
                self.updates.append((p.param_whole, new_p))
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "momentum": float(self.momentum.get_value()),
                "decay": float(self.decay.get_value()),
                "nesterov": self.nesterov}


class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        super(RMSprop, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = shared_scalar(lr)
        self.rho = shared_scalar(rho)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        self.updates = []

        for p, g, a, c in zip(params, grads, accumulators, constraints):
            new_a = self.rho * a + (1 - self.rho) * g ** 2  # update accumulator
            self.updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            self.updates.append((p, c(new_p)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "rho": float(self.rho.get_value()),
                "epsilon": self.epsilon}


class Adagrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = shared_scalar(lr)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        self.updates = []

        for p, g, a, c in zip(params, grads, accumulators, constraints):
            new_a = a + g ** 2  # update accumulator
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            self.updates.append((p, c(new_p)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "epsilon": self.epsilon}


class Adadelta(Optimizer):
    '''
        Reference: http://arxiv.org/abs/1212.5701
    '''
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, *args, **kwargs):
        super(Adadelta, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = shared_scalar(lr)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        delta_accumulators = [shared_zeros(p.get_value().shape) for p in params]
        self.updates = []

        for p, g, a, d_a, c in zip(params, grads, accumulators,
                                   delta_accumulators, constraints):
            new_a = self.rho * a + (1 - self.rho) * g ** 2  # update accumulator
            self.updates.append((a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * T.sqrt(d_a + self.epsilon) / T.sqrt(new_a +
                                                             self.epsilon)

            new_p = p - self.lr * update
            self.updates.append((p, c(new_p)))  # apply constraints

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * update ** 2
            self.updates.append((d_a, new_d_a))
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "rho": self.rho,
                "epsilon": self.epsilon}


class Adam(Optimizer):
    '''
        Reference: http://arxiv.org/abs/1412.6980v8

        Default parameters follow those provided in the original paper.
    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, lambda_=1.0-1.0e-8, epsilon=1e-8, *args, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0.)
        self.lr = shared_scalar(lr)
        self.log_beta_1 = shared_scalar(np.log(self.beta_1))
        self.log_lambda = T.as_tensor_variable(np.cast[float_t](np.log(lambda_)))

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations+1.),
                        (self.log_beta_1, self.log_beta_1 + self.log_lambda)]

        t = self.iterations + 1
        beta_1 = T.exp(self.log_beta_1)
        # lr_t = self.lr * T.sqrt(1-self.beta_2**t)/(1-self.beta_1**t)
        lr_t = self.lr * T.sqrt(1.-self.beta_2**t)/(1.-beta_1**t)

        for p, g, c in zip(params, grads, constraints):
            try:
                m = theano.shared(p.get_value() * 0.)  # zero init of moment
                v = theano.shared(p.get_value() * 0.)  # zero init of velocity
                m_t = (beta_1 * m) + (1. - beta_1) * g
                # m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * (g**2)
                p_t = p - lr_t * m_t / (T.sqrt(v_t) + self.epsilon)
                self.updates.append((m, m_t))
                self.updates.append((v, v_t))
                self.updates.append((p, c(p_t)))  # apply constraints
            except AttributeError:
                # subtensor update:
                # out of the subtensor.
                sub_param_info = p.subtensor_info
                param_shape = sub_param_info.shape
                selection = sub_param_info.idx
                param = sub_param_info.base

                m = theano.shared(np.zeros(shape=param_shape, dtype=float_t), borrow=True)
                v = theano.shared(np.zeros(shape=param_shape, dtype=float_t), borrow=True)
                # lb1_scalar = T.log(self.beta_1)
                # lb2_scalar = T.log(self.beta_2)

                # lb1_scalar = T.as_tensor_variable(np.cast[float_t](np.log(self.beta_1)))
                lb1_scalar = self.log_beta_1
                lb2_scalar = T.as_tensor_variable(np.cast[float_t](np.log(self.beta_2)))
                log_beta_1_ = theano.shared(np.log(self.beta_1) * np.ones(shape=(param_shape[0], ), dtype=float_t))
                log_beta_2_ = theano.shared(np.log(self.beta_2) * np.ones(shape=(param_shape[0], ), dtype=float_t))
                log_beta_1 = log_beta_1_.dimshuffle(0, 'x')
                log_beta_2 = log_beta_2_.dimshuffle(0, 'x')
                sub_momentum = m[selection]
                sub_volocity = v[selection]
                sub_log_beta_1 = log_beta_1[selection]
                sub_log_beta_2 = log_beta_2[selection]

                sub_beta_1 = T.exp(sub_log_beta_1)
                sub_beta_2 = T.exp(sub_log_beta_2)

                m_t = (sub_beta_1 * sub_momentum) + (1. - T.exp(self.log_beta_1)) * g
                v_t = (sub_beta_2 * sub_volocity) + (1. - self.beta_2) * (g**2)
                p_t = T.set_subtensor(p,  c(p - lr_t * m_t / (T.sqrt(v_t) + self.epsilon)))
                self.updates.append((m, T.set_subtensor(sub_momentum, m_t)))
                self.updates.append((v, T.set_subtensor(sub_volocity, v_t)))
                self.updates.append((param, p_t))

                lb1 = log_beta_1.squeeze() + lb1_scalar + self.log_lambda
                lb2 = log_beta_2.squeeze() + lb2_scalar
                new_lb1 = T.set_subtensor(lb1[selection], lb1_scalar)
                new_lb2 = T.set_subtensor(lb2[selection], lb2_scalar)
                self.updates.append((log_beta_1_, new_lb1))
                self.updates.append((log_beta_2_, new_lb2))

                # beta1_scalar = T.as_tensor_variable(np.cast[float_t](self.beta_1))
                # beta2_scalar = T.as_tensor_variable(np.cast[float_t](self.beta_2))
                # beta_1_ = theano.shared(self.beta_1 * np.ones(shape=(param_shape[0], ), dtype=float_t))
                # beta_2_ = theano.shared(self.beta_2 * np.ones(shape=(param_shape[0], ), dtype=float_t))
                # beta_1 = beta_1_.dimshuffle(0, 'x')
                # beta_2 = beta_2_.dimshuffle(0, 'x')
                # sub_momentum = m[selection]
                # sub_volocity = v[selection]
                # sub_beta_1 = beta_1[selection]
                # sub_beta_2 = beta_2[selection]
                #
                # m_t = (sub_beta_1 * sub_momentum) + (1. - self.beta_1) * g
                # v_t = (sub_beta_2 * sub_volocity) + (1. - self.beta_2) * (g**2)
                # p_t = T.set_subtensor(p,  c(p - lr_t * m_t / (T.sqrt(v_t) + self.epsilon)))
                # self.updates.append((m, T.set_subtensor(sub_momentum, m_t)))
                # self.updates.append((v, T.set_subtensor(sub_volocity, v_t)))
                # self.updates.append((param, p_t))
                #
                # lb1 = beta_1.squeeze() * beta1_scalar
                # lb2 = beta_2.squeeze() * beta2_scalar
                # new_lb1 = T.set_subtensor(lb1[selection], beta1_scalar)
                # new_lb2 = T.set_subtensor(lb2[selection], beta2_scalar)
                # self.updates.append((beta_1_, new_lb1))
                # self.updates.append((beta_2_, new_lb2))

        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon}

# aliases
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer', instantiate=True,
                           kwargs=kwargs)
