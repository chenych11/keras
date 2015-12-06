# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

# import theano.tensor as T
from theano.tensor import TensorType
from ..layers.core import Layer, LayerList
from ..utils.theano_utils import ndim_tensor
# noinspection PyUnresolvedReferences
from six.moves import range

import logging

logger = logging.getLogger('keras.layers.containers')


class Sequential(Layer):
    """
        Simple linear stack of layers.

        inherited from Layer:
        - get_params
        - get_output_mask
        - supports_masked_input
    """

    def __init__(self, layers=()):
        super(Sequential, self).__init__()
        self.layers = []
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        self.__input_slots = [None]
        self.__output_slots = [None]

        for layer in layers:
            self.add(layer)

    @property
    def nb_input(self):
        return 1

    @property
    def nb_output(self):
        return 1

    # noinspection PyMethodOverriding
    def set_previous(self, layer):
        """
        :param layer: a list/tuple of Layers
        """
        self.layers[0].previous = layer

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
            if not hasattr(self.layers[0], 'input'):
                self.set_input()
        else:
            layer.on_connection_end()
        layer.init_updates()

        params, regularizers, constraints, updates = layer.get_params()
        self.params += params
        self.regularizers += regularizers
        self.constraints += constraints
        self.updates += updates

    def get_output(self, train=False):
        return self.layers[-1].get_output(train)

    def set_input(self):
        for l in self.layers:
            if hasattr(l, 'input'):
                ndim = l.input.ndim
                self.layers[0].input = ndim_tensor(ndim)
                break

    def set_input_slots(self, inputs):
        """
        :param inputs: a map from bool to theano.tensor variable.
        :type inputs: dict
        :return: None
        """
        self.__input_slots[0] = inputs

    def set_output_slots(self):
        self.__output_slots[0] = {False: self.get_output(train=False), True: self.get_output(train=True)}

    def get_input(self, train=False):
        if not hasattr(self.layers[0], 'input'):
            self.set_input()
        return self.layers[0].get_input(train)

    @property
    def input(self):
        return self.get_input()

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].params)
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "layers": [layer.get_config() for layer in self.layers]}

    def count_params(self):
        return sum([layer.count_params() for layer in self.layers])


class Graph(Layer):
    """
        Implement a NN graph with arbitrary layer connections,
        arbitrary number of inputs and arbitrary number of outputs.

        Note: Graph can only be used as a layer
        (connect, input, get_input, get_output)
        when it has exactly one input and one output.

        inherited from Layer:
            - get_params
            - get_output_mask
            - supports_masked_input
            - get_weights
            - set_weights
    """
    def __init__(self):
        super(Graph, self).__init__()
        self.namespace = set()    # strings
        self.nodes = {}           # layer-like
        self.inputs = {}          # layer-like
        self.input_order = []     # strings
        self.outputs = {}         # layer-like
        self.output_order = []    # strings
        self.input_config = []    # dicts
        self.output_config = []   # dicts
        self.node_config = []     # dicts

        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

        self.__input_slots = None
        self.__output_slots = None

    @property
    def nb_input(self):
        return len(self.inputs)

    @property
    def nb_output(self):
        return len(self.outputs)

    def set_previous(self, layer, connection_map=None):
        if self.nb_input != layer.nb_output:
            raise Exception('Cannot connect layers: input count does not match output count.')
        if self.nb_input == 1:
            self.inputs[self.input_order[0]].set_previous(layer)
        else:
            if not connection_map:
                raise Exception('Cannot attach multi-input layer: no connection_map provided.')
            for k, v in connection_map.items():
                if k in self.inputs and v in layer.outputs:
                    self.inputs[k].set_previous(layer.outputs[v])
                else:
                    raise Exception('Invalid connection map.')

    def get_input(self, train=False):
        if len(self.inputs) == len(self.outputs) == 1:
            return self.inputs[self.input_order[0]].get_input(train)
        else:
            return dict([(k, v.get_input(train)) for k, v in self.inputs.items()])

    @property
    def input(self):
        return self.get_input()

    def get_output(self, train=False):
        if len(self.inputs) == len(self.outputs) == 1:
            return self.outputs[self.output_order[0]].get_output(train)
        else:
            return dict([(k, v.get_output(train)) for k, v in self.outputs.items()])

    def add_input(self, name, ndim=2, dtype='float'):
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self.namespace.add(name)
        self.input_order.append(name)
        layer = Layer()  # empty layer
        if dtype == 'float':
            layer.input = ndim_tensor(ndim)
        else:
            # if ndim == 2:
            #     layer.input = T.imatrix()
            # else:
            #     raise Exception('Type "int" can only be used with ndim==2 (Embedding).')
            tensor_var = TensorType(dtype, (False,)*ndim)
            layer.input = tensor_var()

        layer.input.name = name
        self.inputs[name] = layer
        self.input_config.append({'name': name, 'ndim': ndim, 'dtype': dtype})

    def add_layerlist(self, layerlist, names, inputs):
        if len(names) != layerlist.nb_output:
            logger.warn('Not enough names for each output layer of this layerlist. '
                        'The %dth-%dth layers are dropped.' % (len(names)+1, len(layerlist.output_layers)+1))
        inputs = self.get_nodes(inputs)
        layerlist.set_inputs(inputs)
        for name_, layer_ in zip(names, layerlist.output_layers):
            if name_:
                self.add_node(layer_, name_)

    def get_nodes(self, names):
        if isinstance(names, str):
            names = [names]
        nodes = []
        for n in names:
            if n in self.nodes:
                nodes.append(self.nodes[n])
            elif n in self.inputs:
                nodes.append(self.inputs[n])
            else:
                raise ValueError('Unknown node/input identifier: ' + n)
        return nodes

    def add_node(self, layer, name, inputs=None):
        if isinstance(layer, LayerList):
            self.add_layerlist(layer, name, inputs)
            return

        if hasattr(layer, 'set_name'):
            layer.set_name(name)
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)

        if isinstance(inputs, str):
            inputs = [inputs]

        if inputs is None:
            inputs = ()

        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 1:
                inputs_ = inputs[0]
                if inputs_ not in self.namespace:
                    raise ValueError('Unknown node/input identifier: ' + inputs_)
                if inputs_ in self.nodes:
                    layer.set_previous(self.nodes[inputs_])
                elif inputs_ in self.inputs:
                    layer.set_previous(self.inputs[inputs_])
                else:
                    # should never go here.
                    raise ValueError('%s is in namespace, but not in node set or input set.'
                                     'This indicates the program has a bug.' % inputs_)
            elif len(inputs) == 0:
                pass
            else:
                ilyers = []
                for ilyer_name in inputs:
                    if ilyer_name in self.nodes:
                        ilyers.append(self.nodes[ilyer_name])
                    elif ilyer_name in self.inputs:
                        ilyers.append(self.inputs[ilyer_name])
                layer.set_previous(ilyers)
        else:
            raise TypeError('Only accept str, list or tuple as inputs')

        self.namespace.add(name)
        self.nodes[name] = layer
        self.node_config.append({'name': name, 'inputs': inputs})

        layer.init_updates()
        params, regularizers, constraints, updates = layer.get_params()
        self.params += params
        self.regularizers += regularizers
        self.constraints += constraints
        self.updates += updates

    def add_output(self, name, node=None):
        """ Mark a node or input node as output node.
        :param name: output name.
        :param node: the name of the node to be marked
        :return: None
        """
        if name in self.output_order:
            raise Exception('Duplicate output identifier: ' + name)

        if node not in self.namespace:
            raise ValueError('Unknown node/input identifier: ' + node)
        if node in self.nodes:
            self.outputs[name] = self.nodes[node]
        elif node in self.inputs:
            self.outputs[name] = self.inputs[node]
        else:
            raise ValueError('%s is in namespace, but not in node set or input set.'
                             'This indicates the program has a bug.' % node)
        self.output_order.append(name)
        self.output_config.append({'name': name, 'inputs': node})

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_config": self.input_config,
                "node_config": self.node_config,
                "output_config": self.output_config,
                "input_order": self.input_order,
                "output_order": self.output_order,
                "nodes": dict([(c["name"], self.nodes[c["name"]].get_config()) for c in self.node_config])}

    def count_params(self):
        return sum([layer.count_params() for layer in self.nodes.values()])
