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
from tensorforce.core import tf_util
from tensorforce.core.parameters import Parameter


class PiecewiseConstant(Parameter):
    """
    Piecewise-constant hyperparameter (specification key: `piecewise_constant`).

    Args:
        unit ("timesteps" | "episodes" | "updates"): Unit of interval boundaries
            (<span style="color:#C00000"><b>required</b></span>).
        boundaries (iter[long]): Strictly increasing interval boundaries for constant segments
            (<span style="color:#C00000"><b>required</b></span>).
        values (iter[dtype-dependent]): Interval values of constant segments, one more than
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        dtype (type): <span style="color:#0000C0"><b>internal use</b></span>.
        min_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
        max_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, unit, boundaries, values, name=None, dtype=None, min_value=None, max_value=None
    ):
        if isinstance(values[0], bool):
            if dtype != 'bool':
                raise TensorforceError.unexpected()
        elif isinstance(values[0], int):
            if dtype != 'int':
                raise TensorforceError.unexpected()
        elif isinstance(values[0], float):
            if dtype != 'float':
                raise TensorforceError.unexpected()
        else:
            raise TensorforceError.unexpected()

        assert unit in ('timesteps', 'episodes', 'updates')
        assert len(values) == len(boundaries) + 1
        assert boundaries == sorted(boundaries) and boundaries[0] > 0
        assert all(isinstance(value, type(values[0])) for value in values)

        self.boundaries = boundaries
        self.values = values

        super().__init__(
            unit=unit, name=name, dtype=dtype, min_value=min_value, max_value=max_value
        )

    def min_value(self):
        return min(self.values)

    def max_value(self):
        return max(self.values)

    def final_value(self):
        return self.values[-1]

    def parameter_value(self, *, step):
        parameter = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=self.boundaries, values=self.values
        )(step=step)

        parameter = tf_util.cast(x=parameter, dtype=self.spec.type)

        return parameter
