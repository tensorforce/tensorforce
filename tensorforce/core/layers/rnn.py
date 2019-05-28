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

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core.layers import TransformationBase


class Rnn(TransformationBase):
    """
    Recurrent neural network layer (specification key: `rnn`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        cell ('gru' | 'lstm'): The recurrent cell type
            (<span style="color:#C00000"><b>required</b></span>).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        return_final_state (bool): Whether to return the final state instead of the per-step
            outputs (<span style="color:#00C000"><b>default</b></span>: true).
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
        kwargs: Additional arguments for Keras RNN layer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__.
    """

    def __init__(
        self, name, cell, size, return_final_state=True, bias=False, activation=None, dropout=0.0,
        is_trainable=True, input_spec=None, summary_labels=None, l2_regularization=None, **kwargs
    ):
        self.cell = cell
        self.return_final_state = return_final_state

        if self.return_final_state and self.cell == 'lstm':
            assert size % 2 == 0
            self.size = size // 2
        else:
            self.size = size

        if self.cell == 'gru':
            self.rnn = tf.keras.layers.GRU(
                units=self.size, return_sequences=True, return_state=True, name='rnn',
                input_shape=input_spec['shape'], **kwargs  # , dtype=util.tf_dtype(dtype='float')
            )
        elif self.cell == 'lstm':
            self.rnn = tf.keras.layers.LSTM(
                units=self.size, return_sequences=True, return_state=True, name='rnn',
                input_shape=input_spec['shape'], **kwargs  # , dtype=util.tf_dtype(dtype='float')
            )
        else:
            raise TensorforceError.unexpected()

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            is_trainable=is_trainable, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        if self.squeeze and self.return_final_state:
            raise TensorforceError(
                "Invalid combination for Lstm layer: size=0 and return_final_state=True."
            )

    def default_input_spec(self):
        return dict(type='float', shape=(-1, 0))

    def get_output_spec(self, input_spec):
        if self.return_final_state:
            input_spec['shape'] = input_spec['shape'][:-2] + (self.size,)
        elif self.squeeze:
            input_spec['shape'] = input_spec['shape'][:-1]
        else:
            input_spec['shape'] = input_spec['shape'][:-1] + (self.size,)
        input_spec.pop('min_value', None)
        input_spec.pop('max_value', None)

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        self.rnn.build(input_shape=((None,) + self.input_spec['shape']))

        for variable in self.rnn.trainable_weights:
            name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
            self.variables[name] = variable
            if self.is_trainable:
                self.trainable_variables[name] = variable
        for variable in self.rnn.non_trainable_weights:
            name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
            self.variables[name] = variable

    def tf_regularize(self):
        regularization_loss = super().tf_regularize()

        if len(self.rnn.losses) > 0:
            regularization_loss += tf.math.add_n(inputs=self.rnn.losses)

        return regularization_loss

    def tf_apply(self, x, sequence_length=None):
        x = self.rnn(inputs=x, initial_state=None)

        if self.return_final_state:
            if self.cell == 'gru':
                x = x[1]
            elif self.cell == 'lstm':
                x = tf.concat(values=(x[1], x[2]), axis=1)
        else:
            x = x[0]

        return super().tf_apply(x=x)


class Gru(Rnn):
    """
    Gated recurrent unit layer (specification key: `gru`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        cell ('gru' | 'lstm'): The recurrent cell type
            (<span style="color:#C00000"><b>required</b></span>).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        return_final_state (bool): Whether to return the final state instead of the per-step
            outputs (<span style="color:#00C000"><b>default</b></span>: true).
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
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU>`__.
    """

    def __init__(
        self, name, size, return_final_state=True, bias=False, activation=None, dropout=0.0,
        is_trainable=True, input_spec=None, summary_labels=None, l2_regularization=None, **kwargs
    ):
        super().__init__(
            name=name, cell='gru', size=size, return_final_state=return_final_state, bias=bias,
            activation=activation, dropout=dropout, input_spec=input_spec,
            summary_labels=summary_labels, l2_regularization=l2_regularization, **kwargs
        )


class Lstm(Rnn):
    """
    Long short-term memory layer (specification key: `lstm`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        cell ('gru' | 'lstm'): The recurrent cell type
            (<span style="color:#C00000"><b>required</b></span>).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        return_final_state (bool): Whether to return the final state instead of the per-step
            outputs (<span style="color:#00C000"><b>default</b></span>: true).
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
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>`__.
    """

    def __init__(
        self, name, size, return_final_state=True, bias=False, activation=None, dropout=0.0,
        is_trainable=True, input_spec=None, summary_labels=None, l2_regularization=None, **kwargs
    ):
        super().__init__(
            name=name, cell='lstm', size=size, return_final_state=return_final_state, bias=bias,
            activation=activation, dropout=dropout, input_spec=input_spec,
            summary_labels=summary_labels, l2_regularization=l2_regularization, **kwargs
        )
