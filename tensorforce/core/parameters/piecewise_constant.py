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
from tensorforce.core import Module
from tensorforce.core.parameters import Parameter


class PiecewiseConstant(Parameter):
    """
    Piecewise-constant hyperparameter.

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        dtype ("bool" | "int" | "long" | "float"): Tensor type
            (<span style="color:#C00000"><b>required</b></span>).
        unit ("timesteps" | "episodes" | "updates"): Unit of interval boundaries
            (<span style="color:#C00000"><b>required</b></span>).
        boundaries (iter[long]): Strictly increasing interval boundaries for constant segments
            (<span style="color:#C00000"><b>required</b></span>).
        values (iter[dtype-dependent]): Interval values of constant segments, one more than
            (<span style="color:#C00000"><b>required</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, dtype, unit, boundaries, values, summary_labels=None):
        if isinstance(values[0], bool):
            if dtype != 'bool':
                raise TensorforceError.unexpected()
        elif isinstance(values[0], int):
            if dtype not in ('int', 'long'):
                raise TensorforceError.unexpected()
        elif isinstance(values[0], float):
            if dtype != 'float':
                raise TensorforceError.unexpected()
        else:
            raise TensorforceError.unexpected()

        super().__init__(name=name, dtype=dtype, unit=unit, summary_labels=summary_labels)

        assert unit in ('timesteps', 'episodes', 'updates')
        assert len(values) == len(boundaries) + 1
        assert all(isinstance(value, type(values[0])) for value in values)

        self.boundaries = boundaries
        self.values = values

    def get_parameter_value(self, step):
        parameter = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=self.boundaries, values=self.values
        )(step=step)

        if util.dtype(x=parameter) != self.dtype:
            parameter = tf.dtypes.cast(x=parameter, dtype=util.tf_dtype(dtype=self.dtype))

        return parameter
