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

from tensorforce import TensorforceError
from tensorforce.core import Module, TensorSpec, tf_function, tf_util


class Parameter(Module):
    """
    Base class for dynamic hyperparameters.

    Args:
        unit ("timesteps" | "episodes" | "updates"): Unit of parameter schedule
            (<span style="color:#00C000"><b>default</b></span>: timesteps).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        dtype (type): <span style="color:#0000C0"><b>internal use</b></span>.
        shape (iter[int > 0]): <span style="color:#0000C0"><b>internal use</b></span>.
        min_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
        max_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, unit='timesteps', summary_labels=None, name=None, dtype=None, shape=(),
        min_value=None, max_value=None
    ):
        super().__init__(name=name, summary_labels=summary_labels)

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

    def final_value(self):
        raise NotImplementedError

    def parameter_value(self, step):
        raise NotImplementedError

    def input_signature(self, function):
        if function == 'value':
            return ()

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=0)
    def value(self):
        if self.unit is None:
            step = None
        elif self.unit == 'timesteps':
            step = self.root.timesteps
        elif self.unit == 'episodes':
            step = self.root.episodes
        elif self.unit == 'updates':
            step = self.root.updates

        parameter = self.parameter_value(step=step)
        parameter = self.add_summary(label='parameters', name=self.name, tensor=parameter)

        assertions = self.spec.tf_assert(
            x=parameter, include_type_shape=True,
            message='Parameter.value: invalid {{issue}} for {name} value.'.format(name=self.name)
        )
        with tf.control_dependencies(control_inputs=assertions):
            return tf_util.identity(input=parameter)
