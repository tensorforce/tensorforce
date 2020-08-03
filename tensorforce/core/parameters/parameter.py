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
from tensorforce.core import Module, SignatureDict, TensorSpec, tf_function, tf_util


class Parameter(Module):
    """
    Base class for dynamic hyperparameters.

    Args:
        unit ("timesteps" | "episodes" | "updates"): Unit of parameter schedule
            (<span style="color:#00C000"><b>default</b></span>: timesteps).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        dtype (type): <span style="color:#0000C0"><b>internal use</b></span>.
        shape (iter[int > 0]): <span style="color:#0000C0"><b>internal use</b></span>.
        min_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
        max_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, unit='timesteps', name=None, dtype=None, shape=(), min_value=None, max_value=None
    ):
        super().__init__(name=name)

        assert unit in (None, 'timesteps', 'episodes', 'updates')
        self.unit = unit

        self.spec = TensorSpec(type=dtype, shape=shape, min_value=min_value, max_value=max_value)

        assert self.min_value() is None or self.max_value() is None or \
            self.min_value() <= self.max_value()
        if self.spec.min_value is not None:
            if self.min_value() is None:
                raise TensorforceError.value(
                    name=self.name, argument='lower bound', value=self.min_value(),
                    hint=('not >= {}'.format(self.spec.min_value))
                )
            elif self.min_value() < self.spec.min_value:
                raise TensorforceError.value(
                    name=self.name, argument='lower bound', value=self.min_value(),
                    hint=('< {}'.format(self.spec.min_value))
                )
        if self.spec.max_value is not None:
            if self.max_value() is None:
                raise TensorforceError.value(
                    name=self.name, argument='upper bound', value=self.max_value(),
                    hint=('not <= {}'.format(self.spec.max_value))
                )
            elif self.max_value() > self.spec.max_value:
                raise TensorforceError.value(
                    name=self.name, argument='upper bound', value=self.max_value(),
                    hint=('> {}'.format(self.spec.max_value))
                )

    def min_value(self):
        return None

    def max_value(self):
        return None

    def is_constant(self, *, value=None):
        if value is None:
            if self.min_value() is not None and self.min_value() == self.max_value():
                assert self.final_value() == self.min_value()
                assert isinstance(self.final_value(), self.spec.py_type())
                return self.final_value()
            else:
                return None
        else:
            assert isinstance(value, self.spec.py_type())
            if self.min_value() == value and self.max_value() == value:
                assert self.final_value() == value
                return True
            else:
                return False

    def final_value(self):
        raise NotImplementedError

    def initialize(self):
        super().initialize()

        self.register_summary(label='parameters', name=('parameters/' + self.name))

    def input_signature(self, *, function):
        if function == 'value':
            return SignatureDict()

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'value':
            return SignatureDict(singleton=self.spec.signature(batched=False))

        else:
            return super().output_signature(function=function)

    def parameter_value(self, *, step):
        raise NotImplementedError

    @tf_function(num_args=0)
    def value(self):
        if self.unit is None:
            step = None
        else:
            step = self.root.units[self.unit]

        parameter = self.parameter_value(step=step)

        dependencies = self.spec.tf_assert(
            x=parameter, include_type_shape=True,
            message='Parameter.value: invalid {{issue}} for {name} value.'.format(name=self.name)
        )

        name = 'parameters/' + self.name
        if self.unit is None:
            step = 'timesteps'
        else:
            step = self.unit
        dependencies.extend(self.summary(label='parameters', name=name, data=parameter, step=step))

        with tf.control_dependencies(control_inputs=dependencies):
            return tf_util.identity(input=parameter)
