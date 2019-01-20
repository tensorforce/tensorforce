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

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core.layers import TransformationBase


class InternalLstm(TransformationBase):
    """
    LSTM layer for internal state.
    """

    def __init__(
        self, name, size, is_initial_state_trainable=False, bias=False, activation=None,
        dropout=None, input_spec=None, l2_regularization=None, summary_labels=None
    ):
        """
        LSTM layer for internal state.

        Args:
            size: LSTM size

        """
        self.is_initial_state_trainable = is_initial_state_trainable

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            input_spec=input_spec, l2_regularization=0.0, summary_labels=summary_labels
        )

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

    def internals_spec(self):
        specification = super().internals_spec()

        if 'state' in specification:
            raise TensorforceError.unexpected()

        specification['state'] = dict(type='float', shape=(2, self.size), batched=True)

        return specification

    def internals_init(self):
        initialization = super().internals_init()

        initialization['state'] = self.initial_state

        return initialization

    def tf_initialize(self):
        super().tf_initialize()

        self.cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.size, name='cell', dtype=util.tf_dtype(dtype='float')
        )
        self.cell.build(input_shape=self.input_spec['shape'][0])

        for variable in self.cell.trainable_weights:
            name = variable.name[variable.name.rindex('cell/') + 5: -2]
            self.variables[name] = variable
            self.trainable_variables[name] = variable
        for variable in self.cell.non_trainable_weights:
            name = variable.name[variable.name.rindex('cell/') + 5: -2]
            self.variables[name] = variable

        self.initial_state = self.add_variable(
            name='initial-state', dtype='float', shape=(2, self.size),
            is_trainable=self.is_initial_state_trainable, initializer='zeros'
        )

    def tf_apply(self, x, state):
        state = tf.contrib.rnn.LSTMStateTuple(c=state[:, 0, :], h=state[:, 1, :])

        x, state = self.cell(inputs=x, state=state)
        state = tf.stack(values=(state.c, state.h), axis=1)

        internals = OrderedDict(state=state)

        return super().tf_apply(x=x), internals


class InternalGru(TransformationBase):
    """
    GRU layer for internal state.
    """

    def __init__(
        self, name, size, is_initial_state_trainable=False, bias=False, activation=None,
        dropout=None, input_spec=None, l2_regularization=None, summary_labels=None
    ):
        """
        GRU layer for internal state.

        Args:
            size: GRU size

        """
        self.is_initial_state_trainable = is_initial_state_trainable

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            input_spec=input_spec, l2_regularization=0.0, summary_labels=summary_labels
        )

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

    def internals_spec(self):
        specification = super().internals_spec()

        if 'state' in specification:
            raise TensorforceError.unexpected()

        specification['state'] = dict(type='float', shape=(self.size,), batched=True)

        return specification

    def internals_init(self):
        initialization = super().internals_init()

        initialization['state'] = self.initial_state

        return initialization

    def tf_initialize(self):
        super().tf_initialize()

        self.cell = tf.nn.rnn_cell.GRUCell(
            num_units=self.size, name='cell', dtype=util.tf_dtype(dtype='float')
        )
        self.cell.build(input_shape=self.input_spec['shape'][0])

        for variable in self.cell.trainable_weights:
            name = variable.name[variable.name.rindex('cell/') + 5: -2]
            self.variables[name] = variable
            self.trainable_variables[name] = variable
        for variable in self.cell.non_trainable_weights:
            name = variable.name[variable.name.rindex('cell/') + 5: -2]
            self.variables[name] = variable

        self.initial_state = self.add_variable(
            name='initial-state', dtype='float', shape=(self.size,),
            is_trainable=self.is_initial_state_trainable, initializer='zeros'
        )

    def tf_apply(self, x, state):
        x, state = self.cell(inputs=x, state=state)

        internals = OrderedDict(state=state)

        return super().tf_apply(x=x), internals
