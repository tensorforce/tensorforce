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
from tensorforce.core import Module, tf_function


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

        spec = dict(type=dtype, shape=shape)
        spec = util.valid_value_spec(value_spec=spec, return_normalized=True)
        self.dtype = spec['type']
        self.shape = spec['shape']

        assert min_value is None or max_value is None or min_value < max_value
        if self.dtype == 'bool':
            if min_value is not None or max_value is not None:
                raise TensorforceError.unexpected()
        elif self.dtype in ('int', 'long'):
            if (min_value is not None and not isinstance(min_value, int)) or \
                    (max_value is not None and not isinstance(max_value, int)):
                raise TensorforceError.unexpected()
        elif self.dtype == 'float':
            if (min_value is not None and not isinstance(min_value, float)) or \
                    (max_value is not None and not isinstance(max_value, float)):
                raise TensorforceError.unexpected()
        else:
            assert False

        assert self.min_value() is None or self.max_value() is None or \
            self.min_value() <= self.max_value()
        if min_value is not None:
            if self.min_value() is None:
                raise TensorforceError.value(
                    name=self.name, argument='lower bound', value=self.min_value(),
                    hint=('not >= ' + str(min_value))
                )
            elif self.min_value() < min_value:
                raise TensorforceError.value(
                    name=self.name, argument='lower bound', value=self.min_value(),
                    hint=('< ' + str(min_value))
                )
        if max_value is not None:
            if self.max_value() is None:
                raise TensorforceError.value(
                    name=self.name, argument='upper bound', value=self.max_value(),
                    hint=('not <= ' + str(max_value))
                )
            elif self.max_value() > max_value:
                raise TensorforceError.value(
                    name=self.name, argument='upper bound', value=self.max_value(),
                    hint=('> ' + str(max_value))
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
        else:
            step = self.global_tensor(name=self.unit)

        parameter = self.parameter_value(step=step)

        parameter = self.add_summary(label='parameters', name=self.name, tensor=parameter)

        self.set_global_tensor(name='value', tensor=parameter)

        return parameter
