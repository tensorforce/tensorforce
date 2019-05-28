# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections import OrderedDict
from math import sqrt

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core.layers import StatefulLayer, TransformationBase


class InternalRnn(StatefulLayer, TransformationBase):
    """
    Internal state RNN cell layer (specification key: `internal_rnn`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        cell ('gru' | 'lstm'): The recurrent cell type
            (<span style="color:#C00000"><b>required</b></span>).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        length (parameter, long > 0): ???+1 (<span style="color:#C00000"><b>required</b></span>).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: false).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: none).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        is_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Additional arguments for Keras RNN cell layer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__.
    """

    def __init__(
        self, name, cell, size, length, bias=False, activation=None, dropout=0.0,
        is_trainable=True, input_spec=None, summary_labels=None, l2_regularization=None, **kwargs
    ):
        self.cell_type = cell

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            is_trainable=is_trainable, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization, optimization_horizon=length
        )

        if self.cell_type == 'gru':
            self.cell = tf.keras.layers.GRUCell(
                units=self.size, name='cell', **kwargs  # , dtype=util.tf_dtype(dtype='float')
            )
        elif self.cell_type == 'lstm':
            self.cell = tf.keras.layers.LSTMCell(
                units=self.size, name='cell', **kwargs  # , dtype=util.tf_dtype(dtype='float')
            )
        else:
            raise TensorforceError.unexpected()

    def default_input_spec(self):
        return dict(type='float', shape=(0,))

    def get_output_spec(self, input_spec):
        if self.squeeze:
            input_spec['shape'] = input_spec['shape'][:-1]
        else:
            input_spec['shape'] = input_spec['shape'][:-1] + (self.size,)
        input_spec.pop('min_value', None)
        input_spec.pop('max_value', None)

        return input_spec

    @classmethod
    def internals_spec(cls, layer=None, cell=None, size=None, **kwargs):
        internals_spec = OrderedDict()

        if 'state' in internals_spec:
            raise TensorforceError.unexpected()

        if layer is None:
            assert cell is not None and size is not None
        else:
            assert cell is None and size is None
            cell = layer.cell_type
            size = layer.size

        if cell == 'gru':
            shape = (size,)
        elif cell == 'lstm':
            shape = (2, size)
        internals_spec['state'] = dict(type='float', shape=shape)

        return internals_spec

    def internals_init(self):
        internals_init = OrderedDict()

        if self.cell_type == 'gru':
            shape = (self.size,)
        elif self.cell_type == 'lstm':
            shape = (2, self.size)

        stddev = min(0.1, sqrt(2.0 / self.size))
        internals_init['state'] = np.random.normal(scale=stddev, size=shape)

        return internals_init

    def tf_initialize(self):
        super().tf_initialize()

        self.cell.build(input_shape=self.input_spec['shape'][0])

        for variable in self.cell.trainable_weights:
            name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
            self.variables[name] = variable
            if self.is_trainable:
                self.trainable_variables[name] = variable
        for variable in self.cell.non_trainable_weights:
            name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
            self.variables[name] = variable

    def tf_iterative_step(self, x, previous):
        state = previous['state']

        if self.cell_type == 'gru':
            state = (state,)
        elif self.cell_type == 'lstm':
            state = (state[:, 0, :], state[:, 1, :])

        if util.tf_dtype(dtype='float') not in (tf.float32, tf.float64):
            x = tf.dtypes.cast(x=x, dtype=tf.float32)
            state = util.fmap(function=(lambda x: tf.dtypes.cast(x=x, dtype=tf.float32)), xs=state)
            state = tf.dtypes.cast(x=state, dtype=tf.float32)

        x, state = self.cell(inputs=x, states=state)

        if util.tf_dtype(dtype='float') not in (tf.float32, tf.float64):
            x = tf.dtypes.cast(x=x, dtype=util.tf_dtype(dtype='float'))
            state = util.fmap(
                function=(lambda x: tf.dtypes.cast(x=x, dtype=util.tf_dtype(dtype='float'))),
                xs=state
            )

        if self.cell_type == 'gru':
            state = state[0]
        elif self.cell_type == 'lstm':
            state = tf.stack(values=state, axis=1)

        return x, OrderedDict(state=state)


class InternalGru(InternalRnn):
    """
    Internal state GRU cell layer (specification key: `internal_gru`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        cell ('gru' | 'lstm'): The recurrent cell type
            (<span style="color:#C00000"><b>required</b></span>).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        length (parameter, long > 0): ???+1 (<span style="color:#C00000"><b>required</b></span>).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: false).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: none).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        is_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Additional arguments for Keras GRU layer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell>`__.
    """

    def __init__(
        self, name, size, bias=False, activation=None, dropout=0.0, is_trainable=True,
        input_spec=None, summary_labels=None, l2_regularization=None, **kwargs
    ):
        super().__init__(
            name=name, cell='gru', size=size, bias=bias, activation=activation, dropout=dropout,
            is_trainable=is_trainable, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization, **kwargs
        )

    @classmethod
    def internals_spec(cls, layer=None, **kwargs):
        if layer is None:
            return super().internals_spec(cell='gru', **kwargs)
        else:
            return super().internals_spec(layer=layer)


class InternalLstm(InternalRnn):
    """
    Internal state LSTM cell layer (specification key: `internal_lstm`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        cell ('gru' | 'lstm'): The recurrent cell type
            (<span style="color:#C00000"><b>required</b></span>).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        length (parameter, long > 0): ???+1 (<span style="color:#C00000"><b>required</b></span>).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: false).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: none).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        is_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Additional arguments for Keras LSTM layer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell>`__.
    """

    def __init__(
        self, name, size, bias=False, activation=None, dropout=0.0, is_trainable=True,
        input_spec=None, summary_labels=None, l2_regularization=None, **kwargs
    ):
        super().__init__(
            name=name, cell='lstm', size=size, bias=bias, activation=activation, dropout=dropout,
            is_trainable=is_trainable, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization, **kwargs
        )

    @classmethod
    def internals_spec(cls, layer=None, **kwargs):
        if layer is None:
            return super().internals_spec(cell='lstm', **kwargs)
        else:
            return super().internals_spec(layer=layer)
