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
from tensorforce.core.layers import Layer, TransformationBase


class InternalLayer(Layer):
    """
    Base class for layer with internal state.
    """

    @classmethod
    def internals_spec(cls, layer=None, **kwargs):
        return OrderedDict()

    def internals_init(self):
        return OrderedDict()

    def tf_apply(self, x, **internals):
        return super().tf_apply(x=x)

    def create_tf_function(self, name, tf_function):
        # if name[-6:] != '.apply':
        if tf_function.__name__ != 'tf_apply':
            return super().create_tf_function(name=name, tf_function=tf_function)

        def validated_tf_function(x, **internals):
            if not util.is_consistent_with_value_spec(value_spec=self.input_spec, x=x):
                raise TensorforceError("Invalid input arguments for tf_apply.")
            if not all(
                util.is_consistent_with_value_spec(value_spec=spec, x=internals[name])
                for name, spec in self.__class__.internals_spec(layer=self).items()
            ):
                raise TensorforceError("Invalid input arguments for tf_apply.")

            x, internals = tf_function(x=x, **internals)

            if not util.is_consistent_with_value_spec(value_spec=self.output_spec, x=x):
                raise TensorforceError("Invalid output arguments for tf_apply.")
            if not all(
                util.is_consistent_with_value_spec(value_spec=spec, x=internals[name])
                for name, spec in self.__class__.internals_spec(layer=self).items()
            ):
                raise TensorforceError("Invalid input arguments for tf_apply.")

            if len(internals) > 0:
                return x, internals
            else:
                return x

        return super().create_tf_function(name=name, tf_function=validated_tf_function)


class InternalLstm(InternalLayer, TransformationBase):
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

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            input_spec=input_spec, l2_regularization=0.0, summary_labels=summary_labels
        )

        self.cell = tf.keras.layers.LSTMCell(
            units=self.size, name='cell', dtype=util.tf_dtype(dtype='float')
        )

        self.is_initial_state_trainable = is_initial_state_trainable

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
    def internals_spec(cls, layer=None, size=None, **kwargs):
        internals_spec = super().internals_spec()

        if 'state' in internals_spec:
            raise TensorforceError.unexpected()

        if layer is None:
            internals_spec['state'] = dict(type='float', shape=(2, size))
        else:
            internals_spec['state'] = dict(type='float', shape=(2, layer.size))

        return internals_spec

    def internals_init(self):
        internals_init = super().internals_init()

        internals_init['state'] = self.initial_state

        return internals_init

    def tf_initialize(self):
        super().tf_initialize()

        self.cell.build(input_shape=self.input_spec['shape'][0])

        for variable in self.cell.trainable_weights:
            name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
            self.variables[name] = variable
            self.trainable_variables[name] = variable
        for variable in self.cell.non_trainable_weights:
            name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
            self.variables[name] = variable

        self.initial_state = self.add_variable(
            name='initial-state', dtype='float', shape=(2, self.size),
            is_trainable=self.is_initial_state_trainable, initializer='zeros'
        )

    def tf_apply(self, x, state):
        states = (state[:, 0, :], state[:, 1, :])
        x, states = self.cell(inputs=x, states=states)
        state = tf.stack(values=states, axis=1)

        return super().tf_apply(x=x), OrderedDict(state=state)


class InternalGru(InternalLayer, TransformationBase):
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
        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            input_spec=input_spec, l2_regularization=0.0, summary_labels=summary_labels
        )

        self.cell = tf.keras.layers.GRUCell(
            units=self.size, name='cell', dtype=util.tf_dtype(dtype='float')
        )

        self.is_initial_state_trainable = is_initial_state_trainable

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
    def internals_spec(cls, layer=None, size=None, **kwargs):
        internals_spec = super().internals_spec()

        if 'state' in internals_spec:
            raise TensorforceError.unexpected()

        if layer is None:
            internals_spec['state'] = dict(type='float', shape=(size,))
        else:
            internals_spec['state'] = dict(type='float', shape=(layer.size,))

        return internals_spec

    def internals_init(self):
        internals_init = super().internals_init()

        internals_init['state'] = self.initial_state

        return internals_init

    def tf_initialize(self):
        super().tf_initialize()

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
        x, state = self.cell(inputs=x, states=state)

        return super().tf_apply(x=x), OrderedDict(state=state)
