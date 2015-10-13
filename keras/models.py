from __future__ import absolute_import
from __future__ import print_function
import warnings
import pprint
import logging

import theano
import theano.tensor as T
import numpy as np
import six

# noinspection PyUnresolvedReferences
from six.moves import range
from . import optimizers
from . import objectives
from . import callbacks as cbks
from .utils.layer_utils import container_from_config
from .utils.generic_utils import Progbar
from .layers import containers


logger = logging.getLogger('keras.models')


def standardize_y(y):
    if not hasattr(y, 'shape'):
        y = np.asarray(y)
    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)
    return y


def batch_shuffle(index_array, batch_size):
    batch_count = int(len(index_array)/batch_size)
    # to reshape we need to be cleanly divisible by batch size
    # we stash extra items and reappend them after shuffling
    last_batch = index_array[batch_count*batch_size:]
    index_array = index_array[:batch_count*batch_size]
    index_array = index_array.reshape((batch_count, batch_size))
    np.random.shuffle(index_array)
    index_array = index_array.ravel()
    return np.append(index_array, last_batch)


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def standardize_X(X):
    if type(X) == list:
        return X
    else:
        return [X]


def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            return X[start]
        else:
            return X[start:stop]


def objective_fnc(fn):
    def symvar(y_true, y_pred, mask=None):
        obj_output = fn(y_true, y_pred)
        if mask is None:
            return obj_output.mean(dtype=theano.config.floatX)
        else:
            obj_output = obj_output[mask.nonzero()]
            return obj_output.mean(dtype=theano.config.floatX)
    return symvar


def weighted_objective(fn):
    def weighted(y_true, y_pred, weights, mask=None):
        # it's important that 0 * Inf == 0, not NaN, so we need to filter
        # those out first
        filtered_y_true = y_true[weights.nonzero()[:-1]]
        filtered_y_pred = y_pred[weights.nonzero()[:-1]]
        filtered_weights = weights[weights.nonzero()]
        obj_output = fn(filtered_y_true, filtered_y_pred)
        weighted_obj = filtered_weights * obj_output
        if mask is None:
            # Instead of calling mean() here, we divide by the sum of filtered_weights.
            return weighted_obj.sum() / filtered_weights.sum()
        else:
            filtered_mask = mask[weights.nonzero()[:-1]]
            return weighted_obj.sum() / (filtered_mask * filtered_weights).sum()
    return weighted


def standardize_weights(y, sample_weight=None, class_weight=None):
    if sample_weight is not None:
        return standardize_y(sample_weight)
    elif isinstance(class_weight, dict):
        if len(y.shape) > 3:
            raise Exception('class_weight not supported for 4+ dimensional targets.')
        yshape = y.shape
        y = np.reshape(y, (-1, yshape[-1]))  # for time-distributed data, collapse time and sample
        if y.shape[1] > 1:
            y_classes = y.argmax(axis=1)
        elif y.shape[1] == 1:
            y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y
        class_weights = np.asarray([class_weight[cls] for cls in y_classes])
        return np.reshape(class_weights, yshape[:-1] + (1,))  # uncollapse initial dimensions
    else:
        return np.ones(y.shape[:-1] + (1,))


def model_from_yaml(yaml_string):
    """
        Returns a model generated from a local yaml file,
        which is either created by hand or from to_yaml method of Sequential or Graph
    """
    import yaml
    config = yaml.load(yaml_string)
    return model_from_config(config)


def model_from_json(json_string):
    import json
    config = json.loads(json_string)
    return model_from_config(config)


def model_from_config(config):
    model_name = config.get('name')
    if model_name not in {'Graph', 'Sequential'}:
        raise Exception('Unrecognized model:', model_name)

    # Create a container then set class to appropriate model
    model = container_from_config(config)
    if model_name == 'Graph':
        model.__class__ = Graph
    elif model_name == 'Sequential':
        model.__class__ = Sequential

    if 'optimizer' in config:
        # if it has an optimizer, the model is assumed to be compiled
        loss = config.get('loss')
        class_mode = config.get('class_mode')
        theano_mode = config.get('theano_mode')

        optimizer_params = dict([(k, v) for k, v in config.get('optimizer').items()])
        optimizer_name = optimizer_params.pop('name')
        optimizer = optimizers.get(optimizer_name, optimizer_params)

        if model_name == 'Sequential':
            model.compile(loss=loss, optimizer=optimizer, class_mode=class_mode, theano_mode=theano_mode)
        elif model_name == 'Graph':
            model.compile(loss=loss, optimizer=optimizer, theano_mode=theano_mode)

    return model


def get_function_name(o):
    if isinstance(o, six.string_types):
        return o
    else:
        return o.__name__


class Model(object):
    def __init__(self):
        self.loss = None
        self.weights = None
        self.all_metrics = None
        self.optimizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # target of model
        self.y = None
        self.class_mode = None
        self.theano_mode = None

        self._train = None
        self._train_with_acc = None
        self._predict = None
        self._test = None
        self._test_with_acc = None

    def _fit(self, f, ins, callbacks, val_f=None, val_ins=None, metrics=(),
             batch_size=128, nb_epoch=100, extra_callbacks=(), shuffle=True, verbose=1):
        """
            Abstract fit function for f(*ins). Assume that f returns a list, labelled by out_labels.
        """
        if f.n_returned_outputs == 0:
            raise ValueError('We can not evaluate the outputs with none outputs')

        standardize_outputs = lambda outputs: [outputs] if f.n_returned_outputs == 1 else outputs
        extra_callbacks = list(extra_callbacks)
        nb_train_sample = len(ins[0])

        # logger.debug('out_labels: %s' % str(f.out_labels))

        do_validation = False
        if val_f and val_ins:
            do_validation = True
            pre_train_info = "Train on %d samples, validate on %d samples" % (len(ins[0]), len(val_ins[0]))
        else:
            pre_train_info = "Train on %d samples." % len(ins[0])

        if verbose:
            logger.info(pre_train_info)

        index_array = np.arange(nb_train_sample)
        #  TODO: any good idea to have history as mandatory callback?
        # There is problems for setting history as mandatory callback, for not all metrics are calculated
        # as the way in the History class. So I deleted this function for now and ask the user to define
        # what the callback is.
        # history = cbks.History()
        # callbacks = [history, cbks.BaseLogger()] + callbacks if verbose else [history] + callbacks
        callbacks_ = callbacks
        callbacks = cbks.CallbackList([callbacks_] + extra_callbacks)

        metrics_ = ['val_'+x for x in metrics] + list(metrics)
        cndt_metrics = [m for m in self.all_metrics if m in metrics_]

        callbacks.set_model(self)
        callbacks.set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': list(cndt_metrics),
        })
        callbacks.on_train_begin()

        self.stop_training = False
        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            epoch_logs = {}
            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    ins_batch = slice_X(ins, batch_ids)
                except TypeError:
                    print('TypeError while preparing batch. \
                        If using HDF5 input data, pass shuffle="batch".\n')
                    raise

                batch_logs = {'batch': batch_index, 'size': len(batch_ids)}
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = standardize_outputs(f(*ins_batch))
                _logs = [(label, value) for label, value in zip(f.out_labels, outs)]
                batch_logs.update(_logs)
                callbacks.on_batch_end(batch_index, batch_logs)

                if batch_index == len(batches) - 1:  # last batch
                    # validation
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = self._test_loop(val_f, val_ins, batch_size=batch_size, verbose=0)
                        val_outs = standardize_outputs(val_outs)
                        _logs = [('val_'+label, value) for label, value in zip(val_f.out_labels, val_outs)]
                        epoch_logs.update(_logs)
                        # logger.debug('\nEpoch logs: %s\n' % str(epoch_logs))

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return callbacks_

    @staticmethod
    def _predict_loop(f, ins, batch_size=128, verbose=0):
        """
            Abstract method to loop over some data in batches.
        """
        nb_sample = len(ins[0])
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(*ins_batch)
            if type(batch_outs) != list:
                batch_outs = [batch_outs]
            if batch_index == 0:
                for batch_out in batch_outs:
                    shape = (nb_sample,) + batch_out.shape[1:]
                    outs.append(np.zeros(shape))

            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_start:batch_end] = batch_out
            if verbose == 1:
                # noinspection PyUnboundLocalVariable
                progbar.update(batch_end)
        return outs

    @staticmethod
    def _test_loop(f, ins, batch_size=128, verbose=0):
        """
            Abstract method to loop over some data in batches.
        """
        progbar = None
        nb_sample = len(ins[0])
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(*ins_batch)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(len(batch_ids))

            if verbose == 1:
                progbar.update(batch_end)

        outs = f.summarize_outputs(outs, batch_info)
        return outs

    def get_config(self, verbose=0):
        # config = super(Model, self).get_config()
        config = {}
        for p in ['class_mode', 'theano_mode']:
            if hasattr(self, p):
                config[p] = getattr(self, p)
        if hasattr(self, 'optimizer'):
            config['optimizer'] = self.optimizer.get_config()
        if hasattr(self, 'loss'):
            if type(self.loss) == dict:
                config['loss'] = dict([(k, get_function_name(v)) for k, v in self.loss.items()])
            else:
                config['loss'] = get_function_name(self.loss)

        if verbose:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(config)
        return config

    def to_yaml(self):
        # dump model configuration to yaml string
        import yaml
        config = self.get_config()
        return yaml.dump(config)

    def to_json(self):
        # dump model configuration to json string
        import json
        config = self.get_config()
        return json.dumps(config)


class Sequential(Model, containers.Sequential):
    """
        Inherits from Model the following methods:
            - _fit
            - _predict
            - _evaluate
        Inherits from containers.Sequential the following methods:
            - __init__
            - add
            - get_output
            - get_input
            - get_weights
            - set_weights
    """

    def __init__(self, layers=()):
        # super(Sequential, self).__init__()
        # self.fit = super(Sequential, self).fit
        Model.__init__(self)
        containers.Sequential.__init__(self, layers)

    def compile(self, optimizer, loss, class_mode="categorical", theano_mode=None, with_weights=False):
        inputs_ = {'optimizer': optimizer, 'loss': loss,
                   'class_mode': class_mode, 'theano_mode': theano_mode}
        if with_weights:
            self.__compile_with_weights(**inputs_)
            self.fit = self.__fit_weighted
        else:
            self.__compile_without_weights(**inputs_)
            # noinspection PyAttributeOutsideInit
            self.fit = self.__fit_unweighted

    def __compile_with_weights(self, optimizer, loss, class_mode="categorical", theano_mode=None):
        self.optimizer = optimizers.get(optimizer)

        self.loss = objectives.get(loss)
        weighted_loss = weighted_objective(objectives.get(loss))

        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)

        # target of model
        self.y = T.zeros_like(self.y_train)

        # Fixme: verify this code: I think this code should create a tensor with shape like the y_train
        # except the last dimension, which is set to be one.
        # self.weights = T.ones_like(self.y_train)
        # self.weights = T.ones(self.y_train.shape[:-1] + (1,))
        # weight = T.ones_like(self.y_train).take([0], axis=-1).astype(theano.config.floatX)
        # self.weights = T.unbroadcast(weight, weight.ndim-1)
        TmpTensorType = theano.tensor.TensorType(dtype=theano.config.floatX, broadcastable=(False, )*self.y_train.ndim)
        self.weights = TmpTensorType()

        if hasattr(self.layers[-1], "get_output_mask"):
            mask = self.layers[-1].get_output_mask()
        else:
            mask = None
        train_loss = weighted_loss(self.y, self.y_train, self.weights, mask)
        test_loss = weighted_loss(self.y, self.y_test, self.weights, mask)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        self.y.name = 'y'

        if class_mode == "categorical":
            train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)),
                                    dtype=theano.config.floatX)
            test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)),
                                   dtype=theano.config.floatX)

        elif class_mode == "binary":
            train_accuracy = T.mean(T.eq(self.y, T.round(self.y_train)), dtype=theano.config.floatX)
            test_accuracy = T.mean(T.eq(self.y, T.round(self.y_test)), dtype=theano.config.floatX)
        else:
            raise Exception("Invalid class mode:" + str(class_mode))
        self.class_mode = class_mode
        self.theano_mode = theano_mode

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        if type(self.X_train) == list:
            train_ins = self.X_train + [self.y, self.weights]
            test_ins = self.X_test + [self.y, self.weights]
            predict_ins = self.X_test
        else:
            train_ins = [self.X_train, self.y, self.weights]
            test_ins = [self.X_test, self.y, self.weights]
            predict_ins = [self.X_test]

        self.__compile_fncs(train_ins, train_loss, train_accuracy, test_ins, test_loss, test_accuracy,
                            predict_ins, updates, theano_mode)

        # self._train = theano.function(train_ins, train_loss, updates=updates,
        #                               allow_input_downcast=True, mode=theano_mode)
        # self._train_with_acc = theano.function(train_ins, [train_loss, train_accuracy], updates=updates,
        #                                        allow_input_downcast=True, mode=theano_mode)
        # self._predict = theano.function(predict_ins, self.y_test,
        #                                 allow_input_downcast=True, mode=theano_mode)
        # self._test = theano.function(test_ins, test_loss,
        #                              allow_input_downcast=True, mode=theano_mode)
        # self._test_with_acc = theano.function(test_ins, [test_loss, test_accuracy],
        #                                       allow_input_downcast=True, mode=theano_mode)

    def __compile_without_weights(self, optimizer, loss, class_mode="categorical", theano_mode=None):
        self.optimizer = optimizers.get(optimizer)

        self.loss = objectives.get(loss)
        obj_loss = objective_fnc(self.loss)

        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)

        # target of model
        self.y = T.zeros_like(self.y_train)

        self.weights = None

        if hasattr(self.layers[-1], "get_output_mask"):
            mask = self.layers[-1].get_output_mask()
        else:
            mask = None
        train_loss = obj_loss(self.y, self.y_train, mask)
        test_loss = obj_loss(self.y, self.y_test, mask)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        self.y.name = 'y'

        if class_mode == "categorical":
            train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)),
                                    dtype=theano.config.floatX)
            test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)),
                                   dtype=theano.config.floatX)

        elif class_mode == "binary":
            train_accuracy = T.mean(T.eq(self.y, T.round(self.y_train)), dtype=theano.config.floatX)
            test_accuracy = T.mean(T.eq(self.y, T.round(self.y_test)), dtype=theano.config.floatX)
        else:
            raise Exception("Invalid class mode:" + str(class_mode))
        self.class_mode = class_mode
        self.theano_mode = theano_mode

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        if type(self.X_train) == list:
            train_ins = self.X_train + self.y
            test_ins = self.X_test + self.y
            predict_ins = self.X_test
        else:
            train_ins = [self.X_train, self.y]
            test_ins = [self.X_test, self.y]
            predict_ins = [self.X_test]

        self.__compile_fncs(train_ins, train_loss, train_accuracy, test_ins, test_loss, test_accuracy,
                            predict_ins, updates, theano_mode)

    def __compile_fncs(self, train_ins, train_loss, train_accuracy, test_ins, test_loss, test_accuracy,
                       predict_ins, updates, theano_mode):
        self._train = theano.function(train_ins, train_loss, updates=updates,
                                      allow_input_downcast=True, mode=theano_mode)
        self._train.out_labels = ['loss']

        self._train_with_acc = theano.function(train_ins, [train_loss, train_accuracy], updates=updates,
                                               allow_input_downcast=True, mode=theano_mode)
        self._train_with_acc.out_labels = ['loss', 'acc']

        self._predict = theano.function(predict_ins, self.y_test,
                                        allow_input_downcast=True, mode=theano_mode)
        self._predict.out_labels = ['predicted']

        self._test = theano.function(test_ins, test_loss,
                                     allow_input_downcast=True, mode=theano_mode)
        self._test.out_labels = ['loss']

        self._test_with_acc = theano.function(test_ins, [test_loss, test_accuracy],
                                              allow_input_downcast=True, mode=theano_mode)
        self._test_with_acc.out_labels = ['loss', 'acc']

        self.all_metrics = ['loss', 'acc', 'val_loss', 'val_acc']

        # def __get_metrics_values(f, outs, metrics, prefix=''):
        #     ret = []
        #     out_labels = f.out_labels
        #     metrics = set(metrics)
        #     all_mtrx = set(self.all_metrics)
        #     if not metrics.issubset(all_mtrx):
        #         logger.warn('Specified UNKNOWN metrics ignored')
        #         metrics.difference_update(metrics.difference(all_mtrx))
        #
        #     label2idx = dict((l, idx) for idx, l in enumerate(out_labels))
        #     for mtrx in metrics:
        #         idx = label2idx[mtrx]
        #         ret.append((prefix+mtrx, outs[idx]))
        #     return ret

        def __summary_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)
            return np.sum(out * batch_size, axis=1)/batch_size.sum()

        # self._train_with_acc.get_metrics_values = lambda outs, metrics, prefix='': \
        #     __get_metrics_values(self._train_with_acc, outs, metrics, prefix)
        # self._train.get_metrics_values = lambda outs, metrics, prefix='': \
        #     __get_metrics_values(self._train, outs, metrics, prefix)
        # self._test.get_metrics_values = lambda outs, metrics, prefix='': \
        #     __get_metrics_values(self._test, outs, metrics, prefix)
        # self._test_with_acc.get_metrics_values = lambda outs, metrics, prefix='': \
        #     __get_metrics_values(self._test_with_acc, outs, metrics, prefix)

        self._train_with_acc.summarize_outputs = __summary_outputs
        self._train.summarize_outputs = __summary_outputs
        self._test.summarize_outputs = __summary_outputs
        self._test_with_acc.summarize_outputs = __summary_outputs

    def __prepare_input(self, X, y, class_weight=None, sample_weight=None):
        X = standardize_X(X)
        y = standardize_y(y)
        if self.weights is not None:
            sample_weight = standardize_weights(y, class_weight=class_weight, sample_weight=sample_weight)
            return X + [y, sample_weight]
        else:
            return X + [y]

    def train_on_batch(self, X, y, accuracy=False, class_weight=None, sample_weight=None):
        ins = self.__prepare_input(X, y, class_weight=class_weight, sample_weight=sample_weight)
        if accuracy:
            return self._train_with_acc(*ins)
        else:
            return self._train(*ins)

    def test_on_batch(self, X, y, accuracy=False, sample_weight=None):
        ins = self.__prepare_input(X, y, sample_weight=sample_weight)
        if accuracy:
            return self._test_with_acc(*ins)
        else:
            return self._test(*ins)

    def predict_on_batch(self, X):
        ins = standardize_X(X)
        return self._predict(*ins)

    def __fit_weighted(self, X, y, callbacks, show_metrics, batch_size=128, nb_epoch=100, verbose=1,
                       extra_callbacks=(), validation_split=0., validation_data=None, shuffle=True,
                       show_accuracy=False, class_weight=None, sample_weight=None):

        X = standardize_X(X)
        y = standardize_y(y)

        val_f = None
        val_ins = None
        if validation_data or validation_split:
            if show_accuracy:
                val_f = self._test_with_acc
            else:
                val_f = self._test
        if validation_data:
            if len(validation_data) == 2:
                X_val, y_val = validation_data
                X_val = standardize_X(X_val)
                y_val = standardize_y(y_val)
                sample_weight_val = np.ones(y_val.shape[:-1] + (1,))
            elif len(validation_data) == 3:
                X_val, y_val, sample_weight_val = validation_data
                X_val = standardize_X(X_val)
                y_val = standardize_y(y_val)
                sample_weight_val = standardize_weights(y_val, sample_weight=sample_weight_val)
            else:
                raise Exception("Invalid format for validation data; provide a tuple (X_val, y_val) "
                                "or (X_val, y_val, sample_weight)."
                                "X_val may be a numpy array or a list of numpy arrays depending on your model input.")
            val_ins = X_val + [y_val, sample_weight_val]

        elif 0 < validation_split < 1:
            split_at = int(len(X[0]) * (1 - validation_split))
            X, X_val = (slice_X(X, 0, split_at), slice_X(X, split_at))
            y, y_val = (slice_X(y, 0, split_at), slice_X(y, split_at))
            if sample_weight is not None:
                sample_weight, sample_weight_val = (slice_X(sample_weight, 0, split_at),
                                                    slice_X(sample_weight, split_at))
                sample_weight_val = standardize_weights(y_val, sample_weight=sample_weight_val)
            else:
                sample_weight_val = np.ones(y_val.shape[:-1] + (1,))
            val_ins = X_val + [y_val, sample_weight_val]

        if show_accuracy:
            f = self._train_with_acc
        else:
            f = self._train

        # out_labels = f.out_labels
        sample_weight = standardize_weights(y, class_weight=class_weight, sample_weight=sample_weight)
        ins = X + [y, sample_weight]

        return self._fit(f, ins, callbacks, val_f=val_f, val_ins=val_ins, metrics=show_metrics, batch_size=batch_size,
                         nb_epoch=nb_epoch, extra_callbacks=extra_callbacks, shuffle=shuffle, verbose=verbose)

    def __fit_unweighted(self, X, y, callbacks, show_metrics, batch_size=128, nb_epoch=100, verbose=1,
                         extra_callbacks=(), validation_split=0., validation_data=None,
                         shuffle=True, show_accuracy=False, class_weight=None, sample_weight=None):
        assert self.weights is None
        if class_weight or sample_weight:
            logger.warn('Model compiled without weights. Weights will be dropped.')

        X = standardize_X(X)
        y = standardize_y(y)
        val_f = None
        val_ins = None
        if validation_data or validation_split:
            if show_accuracy:
                val_f = self._test_with_acc
            else:
                val_f = self._test

        if validation_data:
            if len(validation_data) == 2:
                X_val, y_val = validation_data
            else:
                raise Exception("Invalid format for validation data; provide a tuple (X_val, y_val) or (X_val, y_val, "
                                "sample_weight). X_val may be a numpy array or "
                                "a list of numpy arrays depending on your model input.")
            val_ins = self.__prepare_input(X_val, y_val)
        elif 0 < validation_split < 1:
            split_at = int(len(X[0]) * (1 - validation_split))
            X, X_val = (slice_X(X, 0, split_at), slice_X(X, split_at))
            y, y_val = (slice_X(y, 0, split_at), slice_X(y, split_at))
            val_ins = X_val + [y_val]

        if show_accuracy:
            f = self._train_with_acc
            # out_labels = ['loss', 'acc']
        else:
            f = self._train
            # out_labels = ['loss']

        ins = X + [y]
        # logger.debug('Show metrics: %s ' % str(show_metrics))
        return self._fit(f, ins, callbacks, val_f=val_f, val_ins=val_ins, metrics=show_metrics, batch_size=batch_size,
                         nb_epoch=nb_epoch, extra_callbacks=extra_callbacks, shuffle=shuffle, verbose=verbose)

    # noinspection PyMethodMayBeStatic
    def fit(self, X, y, callbacks, show_metrics, batch_size=128, nb_epoch=100, verbose=1, extra_callbacks=(),
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False,
            class_weight=None, sample_weight=None):
        """
        :type X: numpy.ndarray
        :param X: Training feature vectors.
        :type y: numpy.ndarray
        :param y: labels of training data.
        :param batch_size: batch size, default 128
        :type batch_size: int
        :param nb_epoch: number of training epochs.
        :type nb_epoch: int
        :param verbose: verbose output or not
        :param callbacks: a list of callback instances of type keras.callbacks.Callback
        :param validation_split: proportion of validation data to split
        :param validation_data: validation data
        :param shuffle: shuffle or not at each batch/epoch
        :param show_accuracy: show training or validation accuracies or not during training

        :return: callbacks.History
        """
        raise NotImplementedError('This function should be reassigned in the compile function. '
                                  'Please compile the graph first.')

    def predict(self, X, batch_size=128, verbose=0):
        """
        :param X: feature vectors
        :param batch_size: batch size
        :param verbose: verbose or not
        :return: numpy.ndarray
        """
        X = standardize_X(X)
        return self._predict_loop(self._predict, X, batch_size, verbose)[0]

    def predict_proba(self, X, batch_size=128, verbose=1):
        preds = self.predict(X, batch_size, verbose)
        if preds.min() < 0 or preds.max() > 1:
            warnings.warn("Network returning invalid probability values.")
        return preds

    def predict_classes(self, X, batch_size=128, verbose=1):
        proba = self.predict(X, batch_size=batch_size, verbose=verbose)
        if self.class_mode == "categorical":
            return proba.argmax(axis=-1)
        else:
            # return (proba > 0.5).astype('int32')
            return np.greater(proba, 0.5).astype(np.int32)

    def evaluate(self, X, y, batch_size=128, show_accuracy=False, verbose=1, sample_weight=None):
        ins = self.__prepare_input(X, y, sample_weight=sample_weight)
        if show_accuracy:
            f = self._test_with_acc
        else:
            f = self._test
        outs = self._test_loop(f, ins, batch_size, verbose)
        if show_accuracy:
            return outs
        else:
            return outs[0]

    def save_weights(self, filepath, overwrite=False):
        # Save weights from all layers to HDF5
        import h5py
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % filepath)
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        f = h5py.File(filepath, 'w')
        f.attrs['nb_layers'] = len(self.layers)
        for k, l in enumerate(self.layers):
            g = f.create_group('layer_{}'.format(k))
            weights = l.get_weights()
            g.attrs['nb_params'] = len(weights)
            for n, param in enumerate(weights):
                param_name = 'param_{}'.format(n)
                param_dset = g.create_dataset(param_name, param.shape, dtype=param.dtype)
                param_dset[:] = param
        f.flush()
        f.close()

    def load_weights(self, filepath):
        """
            This method does not make use of Sequential.set_weights()
            for backwards compatibility.
        """
        # Loads weights from HDF5 file
        import h5py
        f = h5py.File(filepath)
        for k in range(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            self.layers[k].set_weights(weights)
        f.close()

    def update_model(self):
        layers = self.layers
        self.layers = []
        self.__init__(layers=layers)


# TODO: Test
class Graph(Model, containers.Graph):
    def __init__(self, weighted_inputs=False):
        # super(Graph, self).__init__()
        # todo: check why the code above does not work.
        Model.__init__(self)
        containers.Graph.__init__(self)
        self.is_weighted_input = weighted_inputs

    def compile(self, optimizer, loss, theano_mode=None):
        """
        :param optimizer: Optimizer to choose. See optimizers
        :param loss: a map from output names to loss function names
        :param theano_mode:  the mode to compile the computation graphs. See theano Mode
        :return: None
        """
        if self.is_weighted_input:
            self._compile_with_weights(optimizer, loss, theano_mode)
        else:
            self._compile_without_weights(optimizer, loss, theano_mode)

    def _compile_fncs(self, train_ins, train_loss, updates, test_ins, test_loss, pred_ins, ys_test, theano_mode):
        self._train = theano.function(train_ins, train_loss, updates=updates,
                                      allow_input_downcast=True, mode=theano_mode)
        self._test = theano.function(test_ins, test_loss,
                                     allow_input_downcast=True, mode=theano_mode)
        self._predict = theano.function(inputs=pred_ins, outputs=ys_test,
                                        allow_input_downcast=True, mode=theano_mode)

        self._train.out_labels = ('loss',)
        self._test.out_labels = ('loss', )
        self._predict.out_labels = ('pred',)

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)
            return np.sum(out * batch_size, axis=1)/batch_size.sum()

        self._train.summarize_outputs = __summarize_outputs
        self._test.summarize_outputs = __summarize_outputs

    def _compile_with_weights(self, optimizer, loss, theano_mode=None):
        # loss is a dictionary mapping output name to loss functions
        ys = []
        ys_train = []
        ys_test = []
        weights = []
        train_loss = 0.
        test_loss = 0.
        self.is_weighted_input = True

        for output_name in self.output_order:
            loss_fn = loss[output_name]
            output = self.outputs[output_name]
            y_train = output.get_output(True)
            y_test = output.get_output(False)
            y = T.zeros_like(y_test)
            ys.append(y)
            ys_train.append(y_train)
            ys_test.append(y_test)

            if hasattr(output, "get_output_mask"):
                mask = output.get_output_mask()
            else:
                mask = None

            # Fixme: verify this code: I think this code should create a tensor with shape like the y_train
            # except the last dimension, which is set to be one. Report a bug if verified.
            # One way to solve this is as:
            # weight = T.ones_like(y_test)
            # weight = T.ones_like(y_test).take([0], axis=-1).astype(theano.config.floatX)
            # weight = T.unbroadcast(weight, weight.ndim-1)
            # Another way is simpler: instead of creating a instanced symbolic variable, I just declare a tensor
            # type and then create a symbolic variable of this type.
            TmpTensorType = theano.tensor.TensorType(dtype=theano.config.floatX, broadcastable=(False, )*y_test.ndim)
            weight = TmpTensorType()
            weights.append(weight)
            weighted_loss = weighted_objective(objectives.get(loss_fn))
            train_loss += weighted_loss(y, y_train, weight, mask)
            test_loss += weighted_loss(y, y_test, weight, mask)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'

        ins = [self.inputs[name].input for name in self.input_order]
        train_ins = ins + ys + weights
        test_ins = ins + ys + weights

        for r in self.regularizers:
            train_loss = r(train_loss)
        self.optimizer = optimizers.get(optimizer)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates
        self.theano_mode = theano_mode
        self.loss = loss

        self._compile_fncs(train_ins, train_loss, updates, test_ins, test_loss, ins, ys_test, theano_mode)
        # self._train = theano.function(train_ins, train_loss, updates=updates,
        #                               allow_input_downcast=True, mode=theano_mode)
        # self._test = theano.function(test_ins, test_loss,
        #                              allow_input_downcast=True, mode=theano_mode)
        # self._predict = theano.function(inputs=ins, outputs=ys_test,
        #                                 allow_input_downcast=True, mode=theano_mode)

    def _compile_without_weights(self, optimizer, loss, theano_mode=None):
        # loss is a dictionary mapping output name to loss functions
        ys = []
        ys_train = []
        ys_test = []
        train_loss = 0.
        test_loss = 0.
        self.is_weighted_input = False

        for output_name in self.output_order:
            loss_fn = loss[output_name]
            output = self.outputs[output_name]
            y_train = output.get_output(True)
            y_test = output.get_output(False)
            y = T.zeros_like(y_test)
            ys.append(y)
            ys_train.append(y_train)
            ys_test.append(y_test)

            if hasattr(output, "get_output_mask"):
                mask = output.get_output_mask()
            else:
                mask = None

            unweighted_loss = objective_fnc(objectives.get(loss_fn))
            train_loss += unweighted_loss(y, y_train, mask)
            test_loss += unweighted_loss(y, y_test, mask)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'

        ins = [self.inputs[name].input for name in self.input_order]
        train_ins = ins + ys
        test_ins = ins + ys

        for r in self.regularizers:
            train_loss = r(train_loss)
        self.optimizer = optimizers.get(optimizer)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates
        self.theano_mode = theano_mode
        self.loss = loss

        # self._train = theano.function(train_ins, train_loss, updates=updates,
        #                               allow_input_downcast=True, mode=theano_mode)
        # self._test = theano.function(test_ins, test_loss,
        #                              allow_input_downcast=True, mode=theano_mode)
        # self._predict = theano.function(inputs=ins, outputs=ys_test,
        #                                 allow_input_downcast=True, mode=theano_mode)
        self._compile_fncs(train_ins, train_loss, updates, test_ins, test_loss, ins, ys_test, theano_mode)

    def _prepare_input(self, data, class_weight=None, sample_weight=None):
        if not self.is_weighted_input and (class_weight is not None or sample_weight is not None):
            logger.warn('Compiled without weighted samples (classes) supported. Weights will ignored')
        X = [data[name] for name in self.input_order]
        y = [standardize_y(data[name]) for name in self.output_order]
        if self.is_weighted_input is not None:
            sample_weight = [standardize_weights(data[name], sample_weight=sample_weight.get(name),
                                                 class_weight=class_weight.get(name))
                             for name in self.output_order]
            return X + y + sample_weight
        else:
            return X + y

    def train_on_batch(self, data, class_weight=None, sample_weight=None):
        # class_weight = {} if class_weight is None else class_weight
        # sample_weight = {} if sample_weight is None else sample_weight
        # # data is a dictionary mapping output and input names to arrays
        # sample_weight = [standardize_weights(data[name], sample_weight=sample_weight.get(name),
        #                                      class_weight=class_weight.get(name))
        #                  for name in self.output_order]
        # ins = [data[name] for name in self.input_order] + \
        #       [standardize_y(data[name]) for name in self.output_order] + sample_weight
        ins = self._prepare_input(data, class_weight, sample_weight)
        return self._train(*ins)

    def test_on_batch(self, data, sample_weight=None):
        # sample_weight = {} if sample_weight is None else sample_weight
        # # data is a dictionary mapping input names to arrays
        # sample_weight = [standardize_weights(data[name], sample_weight=sample_weight.get(name))
        #                  for name in self.output_order]
        # ins = [data[name] for name in self.input_order] + \
        #       [standardize_y(data[name]) for name in self.output_order] + sample_weight
        ins = self._prepare_input(data, None, sample_weight)
        return self._test(*ins)

    def predict_on_batch(self, data):
        # data is a dictionary mapping input names to arrays
        ins = [data[name] for name in self.input_order]
        return self._predict(*ins)

    def _fit_unweighted(self, data, callbacks, show_metrics, batch_size=128, nb_epoch=100, verbose=1,
                        extra_callbacks=(), validation_split=0., validation_data=None, shuffle=True,
                        class_weight=None, sample_weight=None):
        # if not self.is_weighted_input and (class_weight is not None or sample_weight is not None):
        #     logger.warn('Compiled without weighted samples (classes) supported. Weights are ignored.')

        # X = [data[name] for name in self.input_order]
        # y = [standardize_y(data[name]) for name in self.output_order]
        ins = self._prepare_input(data, class_weight, sample_weight)

        val_f = None
        val_ins = None
        if validation_data or validation_split:
            val_f = self._test
        if validation_data:
            # val_ins = [validation_data[name] for name in self.input_order] + \
            #           [standardize_y(validation_data[name]) for name in self.output_order]
            val_ins = self._prepare_input(validation_data, sample_weight=sample_weight)

        elif 0 < validation_split < 1:
            split_at = max(int(len(ins[0]) * (1 - validation_split)), 1)
            # X, X_val = (slice_X(X, 0, split_at), slice_X(X, split_at))
            # y, y_val = (slice_X(y, 0, split_at), slice_X(y, split_at))
            # val_ins = X_val + y_val
            ins, val_ins = (self._prepare_input(ins, 0, split_at), self._prepare_input(ins, split_at))

        f = self._train
        # out_labels = ['loss']
        # metrics = ['loss', 'val_loss']

        return self._fit(f, ins, callbacks, val_f=val_f, val_ins=val_ins, metrics=show_metrics,
                         batch_size=batch_size, nb_epoch=nb_epoch, extra_callbacks=extra_callbacks,
                         shuffle=shuffle, verbose=verbose)

    def _fit_weighted(self, data, callbacks, show_metrics, batch_size=128, nb_epoch=100, verbose=1,
                      extra_callbacks=(), validation_split=0., validation_data=None, shuffle=True,
                      class_weight=None, sample_weight=None):
        class_weight = {} if class_weight is None else class_weight
        sample_weight = {} if sample_weight is None else sample_weight

        X = [data[name] for name in self.input_order]
        y = [standardize_y(data[name]) for name in self.output_order]
        sample_weight_list = [standardize_weights(data[name], sample_weight=sample_weight.get(name))
                              for name in self.output_order]
        class_weight_list = [class_weight.get(name) for name in self.output_order]

        val_f = None
        val_ins = None
        if validation_data or validation_split:
            val_f = self._test
        if validation_data:
            # can't use sample weights with validation data at this point
            sample_weight = [standardize_weights(validation_data[name]) for name in self.output_order]
            val_ins = [validation_data[name] for name in self.input_order] + \
                      [standardize_y(validation_data[name]) for name in self.output_order] + sample_weight

        elif 0 < validation_split < 1:
            split_at = max(int(len(X[0]) * (1 - validation_split)), 1)
            X, X_val = (slice_X(X, 0, split_at), slice_X(X, split_at))
            y, y_val = (slice_X(y, 0, split_at), slice_X(y, split_at))
            sample_weight_list, sample_weight_list_val = (slice_X(sample_weight_list, 0, split_at),
                                                          slice_X(sample_weight_list, split_at))
            val_ins = X_val + y_val + sample_weight_list_val

        f = self._train
        # out_labels = ['loss']
        # metrics = ['loss', 'val_loss']

        sample_weight_list = [standardize_weights(y[i], sample_weight=sample_weight_list[i],
                                                  class_weight=class_weight_list[i])
                              for i in range(len(self.output_order))]
        ins = X + y + sample_weight_list

        history = self._fit(f, ins, callbacks, val_f=val_f, val_ins=val_ins, metrics=show_metrics, batch_size=batch_size,
                            nb_epoch=nb_epoch, extra_callbacks=extra_callbacks,  shuffle=shuffle, verbose=verbose)
        return history

    def fit(self, data, callbacks, show_metrics, batch_size=128, nb_epoch=100, verbose=1,
            extra_callbacks=(), validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None):
        kwargs = locals()
        kwargs.pop('self')
        if self.is_weighted_input:
            self._fit_weighted(**kwargs)
        else:
            self._fit_unweighted(**kwargs)

    def evaluate(self, data, batch_size=128, verbose=0, sample_weight=None):
        # sample_weight = {} if sample_weight is None else sample_weight
        # sample_weight = [standardize_weights(data[name], sample_weight=sample_weight.get(name))
        #                  for name in self.output_order]
        #
        # ins = [data[name] for name in self.input_order] + \
        #       [standardize_y(data[name]) for name in self.output_order] + sample_weight
        ins = self._prepare_input(data, sample_weight=sample_weight)
        outs = self._test_loop(self._test, ins, batch_size, verbose)
        # return outs[0]
        return outs

    def predict(self, data, batch_size=128, verbose=0):
        ins = [data[name] for name in self.input_order]
        outs = self._predict_loop(self._predict, ins, batch_size, verbose)
        return dict(zip(self.output_order, outs))

    def save_weights(self, filepath, overwrite=False):
        # Save weights from all layers to HDF5
        import h5py
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % filepath)
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        f = h5py.File(filepath, 'w')
        g = f.create_group('graph')
        weights = self.get_weights()
        g.attrs['nb_params'] = len(weights)
        for n, param in enumerate(weights):
            param_name = 'param_{}'.format(n)
            param_dset = g.create_dataset(param_name, param.shape, dtype=param.dtype)
            param_dset[:] = param
        f.flush()
        f.close()

    def load_weights(self, filepath):
        # Loads weights from HDF5 file
        import h5py
        f = h5py.File(filepath)
        g = f['graph']
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        self.set_weights(weights)
        f.close()
