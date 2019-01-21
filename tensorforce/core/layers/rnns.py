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


class Lstm(TransformationBase):
    """
    LSTM layer.
    """

    def __init__(
        self, name, size, return_final_state=True, bias=False, activation=None, dropout=None,
        input_spec=None, summary_labels=None  # l2_regularization=None
    ):
        """
        LSTM layer.

        Args:
            size: LSTM size (takes into account return_final_state)
            return_final_state: ???
        """
        if return_final_state:
            assert size % 2 == 0
            self.lstm_size = size // 2
        else:
            self.lstm_size = size

        self.return_final_state = return_final_state

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            input_spec=input_spec, l2_regularization=0.0, summary_labels=summary_labels
        )

        if self.squeeze and self.return_final_state:
            raise TensorforceError(
                "Invalid combination for Lstm layer: size=0 and return_final_state=True."
            )

    def default_input_spec(self):
        return dict(type='float', shape=(-1, 0))

    def get_output_spec(self, input_spec):
        if self.return_final_state:
            input_spec['shape'] = input_spec['shape'][:-2] + (2 * self.lstm_size,)
        elif self.squeeze:
            input_spec['shape'] = input_spec['shape'][:-1]
        else:
            input_spec['shape'] = input_spec['shape'][:-1] + (self.lstm_size,)
        input_spec.pop('min_value', None)
        input_spec.pop('max_value', None)

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        self.cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.lstm_size, name='cell', dtype=util.tf_dtype(dtype='float')
        )
        self.cell.build(input_shape=self.input_spec['shape'][1])

        for variable in self.cell.trainable_weights:
            name = variable.name[variable.name.rindex('cell/') + 5: -2]
            self.variables[name] = variable
            self.trainable_variables[name] = variable
        for variable in self.cell.non_trainable_weights:
            name = variable.name[variable.name.rindex('cell/') + 5: -2]
            self.variables[name] = variable

        # if self.lstm_dropout is not None:
        #     keep_prob = self.cond(pred=update, true_fn=(lambda: 1.0 - self.lstm_dropout), false_fn=(lambda: 1.0))
        #     self.lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_cell, output_keep_prob=keep_prob)

    def tf_apply(self, x, sequence_length=None):
        x, state = tf.nn.dynamic_rnn(
            cell=self.cell, inputs=x, sequence_length=sequence_length, initial_state=None,
            dtype=util.tf_dtype(dtype='float'),
            # Weird TensorFlow behavior? (https://github.com/tensorflow/tensorflow/issues/15874)
            parallel_iterations=(self.input_spec['shape'][0] + 1)
        )

        if self.return_final_state:
            x = tf.concat(values=(state.c, state.h), axis=1)

        return super().tf_apply(x=x)


class Gru(TransformationBase):
    """
    GRU layer.
    """

    def __init__(
        self, name, size, return_final_state=True, bias=False, activation=None, dropout=None,
        input_spec=None, summary_labels=None  # l2_regularization=None
    ):
        """
        GRU layer.

        Args:
            size: GRU size (takes into account return_final_state)
            return_final_state: ???
        """
        self.return_final_state = return_final_state

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            input_spec=input_spec, l2_regularization=0.0, summary_labels=summary_labels
        )

        if self.squeeze and self.return_final_state:
            raise TensorforceError(
                "Invalid combination for Gru layer: size=0 and return_final_state=True."
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

        self.cell = tf.nn.rnn_cell.GRUCell(
            num_units=self.size, name='cell', dtype=util.tf_dtype(dtype='float')
        )
        self.cell.build(input_shape=self.input_spec['shape'][1])

        for variable in self.cell.trainable_weights:
            name = variable.name[variable.name.rindex('cell/') + 5: -2]
            self.variables[name] = variable
            self.trainable_variables[name] = variable
        for variable in self.cell.non_trainable_weights:
            name = variable.name[variable.name.rindex('cell/') + 5: -2]
            self.variables[name] = variable

        # if self.lstm_dropout is not None:
        #     keep_prob = self.cond(pred=update, true_fn=(lambda: 1.0 - self.lstm_dropout), false_fn=(lambda: 1.0))
        #     self.lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_cell, output_keep_prob=keep_prob)

    def tf_apply(self, x, sequence_length=None):
        x, state = tf.nn.dynamic_rnn(
            cell=self.cell, inputs=x, sequence_length=sequence_length, initial_state=None,
            dtype=util.tf_dtype(dtype='float'),
            # Weird TensorFlow behavior? (https://github.com/tensorflow/tensorflow/issues/15874)
            parallel_iterations=(self.input_spec['shape'][0] + 1)
        )

        if self.return_final_state:
            x = state

        return super().tf_apply(x=x)
