# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from tensorforce import TensorforceError
from tensorforce.core import TensorSpec, tf_function, tf_util
from tensorforce.core.layers import TransformationBase


class InputRnn(TransformationBase):
    """
    Recurrent neural network layer which is unrolled over a sequence input independently per
    timestep, and consequently does not maintain an internal state (specification key: `input_rnn`).

    Args:
        cell ('gru' | 'lstm'): The recurrent cell type
            (<span style="color:#C00000"><b>required</b></span>).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        return_final_state (bool): Whether to return the final state instead of the per-step
            outputs (<span style="color:#00C000"><b>default</b></span>: true).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: tanh).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
        kwargs: Additional arguments for Keras RNN layer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__.
    """

    def __init__(
        self, *, cell, size, return_final_state=True, bias=True, activation='tanh', dropout=0.0,
        vars_trainable=True, l2_regularization=None, name=None, input_spec=None, **kwargs
    ):
        self.cell_type = cell
        self.return_final_state = return_final_state

        super().__init__(
            size=size, bias=bias, activation=activation, dropout=dropout,
            vars_trainable=vars_trainable, l2_regularization=l2_regularization, name=name,
            input_spec=input_spec
        )

        if self.squeeze and self.return_final_state:
            raise TensorforceError.value(
                name='rnn', argument='return_final_state', value=return_final_state,
                condition='size = 0'
            )

        if self.return_final_state and self.cell_type == 'lstm':
            assert size % 2 == 0
            self.size = size // 2
        else:
            self.size = size

        if self.cell_type == 'gru':
            self.rnn = tf.keras.layers.GRU(
                units=self.size, return_sequences=True, return_state=True, name='rnn',
                input_shape=input_spec.shape, **kwargs  # , dtype=tf_util.get_dtype(type='float')
            )
        elif self.cell_type == 'lstm':
            self.rnn = tf.keras.layers.LSTM(
                units=self.size, return_sequences=True, return_state=True, name='rnn',
                input_shape=input_spec.shape, **kwargs  # , dtype=tf_util.get_dtype(type='float')
            )
        else:
            raise TensorforceError.value(
                name='Rnn', argument='cell', value=self.cell_type, hint='not in {gru,lstm}'
            )

    def default_input_spec(self):
        return TensorSpec(type='float', shape=(-1, 0))

    def output_spec(self):
        output_spec = super().output_spec()

        if self.squeeze:
            output_spec.shape = output_spec.shape[:-1]
        elif not self.return_final_state:
            output_spec.shape = output_spec.shape[:-1] + (self.size,)
        elif self.cell_type == 'gru':
            output_spec.shape = output_spec.shape[:-2] + (self.size,)
        elif self.cell_type == 'lstm':
            output_spec.shape = output_spec.shape[:-2] + (2 * self.size,)

        output_spec.min_value = None
        output_spec.max_value = None

        return output_spec

    def initialize(self):
        super().initialize()

        self.rnn.build(input_shape=((None,) + self.input_spec.shape))

    @tf_function(num_args=0)
    def regularize(self):
        regularization_loss = super().regularize()

        if len(self.rnn.losses) > 0:
            regularization_loss += tf.math.add_n(inputs=self.rnn.losses)

        return regularization_loss

    @tf_function(num_args=1)
    def apply(self, *, x):
        x = tf_util.float32(x=x)
        x = self.rnn(inputs=x, initial_state=None)

        if not self.return_final_state:
            x = tf_util.cast(x=x[0], dtype='float')
        elif self.cell_type == 'gru':
            x = tf_util.cast(x=x[1], dtype='float')
        elif self.cell_type == 'lstm':
            x = tf_util.cast(x=tf.concat(values=(x[1], x[2]), axis=1), dtype='float')

        return super().apply(x=x)


class InputGru(InputRnn):
    """
    Gated recurrent unit layer which is unrolled over a sequence input independently per timestep,
    and consequently does not maintain an internal state (specification key: `input_gru`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        return_final_state (bool): Whether to return the final state instead of the per-step
            outputs (<span style="color:#00C000"><b>default</b></span>: true).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: tanh).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
        kwargs: Additional arguments for Keras GRU layer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU>`__.
    """

    def __init__(
        self, *, size, return_final_state=True, bias=False, activation=None, dropout=0.0,
        vars_trainable=True, l2_regularization=None, name=None, input_spec=None, **kwargs
    ):
        super().__init__(
            cell='gru', size=size, return_final_state=return_final_state, bias=bias,
            activation=activation, dropout=dropout, vars_trainable=vars_trainable,
            l2_regularization=l2_regularization, name=name, input_spec=input_spec, **kwargs
        )


class InputLstm(InputRnn):
    """
    Long short-term memory layer which is unrolled over a sequence input independently per timestep,
    and consequently does not maintain an internal state (specification key: `input_lstm`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        return_final_state (bool): Whether to return the final state instead of the per-step
            outputs (<span style="color:#00C000"><b>default</b></span>: true).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: tanh).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
        kwargs: Additional arguments for Keras LSTM layer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>`__.
    """

    def __init__(
        self, *, size, return_final_state=True, bias=False, activation=None, dropout=0.0,
        vars_trainable=True, l2_regularization=None, name=None, input_spec=None, **kwargs
    ):
        super().__init__(
            cell='lstm', size=size, return_final_state=return_final_state, bias=bias,
            activation=activation, dropout=dropout, vars_trainable=vars_trainable,
            l2_regularization=l2_regularization, name=name, input_spec=input_spec, **kwargs
        )
