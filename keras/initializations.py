from __future__ import absolute_import
import theano
# import theano.tensor as T
import numpy as np
import math

from .utils.theano_utils import sharedX, shared_zeros, shared_ones

DEFAULT_DEVICE = 'tmp'  # use for temporary initialization
floatX = theano.config.floatX


class CPUArray(object):
    def __init__(self, array):
        super(CPUArray, self).__init__()
        self.data = array

    def get_value(self, borrow=True):
        return self.data


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def uniform(shape, scale=0.05, device=DEFAULT_DEVICE):
    w = np.random.uniform(low=-scale, high=scale, size=shape)
    trans = CPUArray if device.lower() == 'cpu' else sharedX
    return trans(w)


def normal(shape, scale=0.05, device=DEFAULT_DEVICE):
    w = np.random.randn(*shape) * scale
    trans = CPUArray if device.lower() == 'cpu' else sharedX
    return trans(w)


def lecun_uniform(shape, device=DEFAULT_DEVICE):
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    fan_in, fan_out = get_fans(shape)
    scale = math.sqrt(3. / fan_in)
    return uniform(shape, scale, device)


def glorot_normal(shape, device=DEFAULT_DEVICE):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    fan_in, fan_out = get_fans(shape)
    s = math.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s, device)


def glorot_uniform(shape, device=DEFAULT_DEVICE):
    fan_in, fan_out = get_fans(shape)
    s = math.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, device)


def he_normal(shape, device=DEFAULT_DEVICE):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in, fan_out = get_fans(shape)
    s = math.sqrt(2. / fan_in)
    return normal(shape, s, device)


def he_uniform(shape, device=DEFAULT_DEVICE):
    fan_in, fan_out = get_fans(shape)
    s = math.sqrt(6. / fan_in)
    return uniform(shape, s, device)


def orthogonal(shape, scale=1.1, device=DEFAULT_DEVICE):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    trans = CPUArray if device.lower() == 'cpu' else sharedX
    return trans(scale * q[:shape[0], :shape[1]])


def identity(shape, scale=1, device=DEFAULT_DEVICE):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Exception("Identity matrix initialization can only be used for 2D square matrices")
    else:
        trans = CPUArray if device.lower() == 'cpu' else sharedX
        return trans(scale * np.identity(shape[0]), device)


def zero(shape, device=DEFAULT_DEVICE):
    if device.lower() == 'cpu':
        return CPUArray(np.zeros(shape))
    return shared_zeros(shape)


def one(shape, device=DEFAULT_DEVICE):
    if device.lower() == 'cpu':
        return CPUArray(np.ones(shape))
    return shared_ones(shape)


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'initialization')
