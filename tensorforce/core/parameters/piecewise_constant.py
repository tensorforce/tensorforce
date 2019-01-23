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

from tensorforce import util
from tensorforce.core import Module
from tensorforce.core.parameters import Parameter


class PiecewiseConstant(Parameter):
    """
    Piecewise constant hyperparameter.
    """

    def __init__(self, name, dtype, unit, boundaries, values, summary_labels=None):
        super().__init__(name=name, dtype=dtype, summary_labels=summary_labels)

        assert unit in ('timesteps', 'episodes')
        assert len(values) == len(boundaries) + 1

        self.unit = unit
        self.boundaries = boundaries
        self.values = values

    def get_parameter_value(self):
        if self.unit == 'timesteps':
            step = Module.retrieve_tensor(name='timestep')
        elif self.unit == 'episodes':
            step = Module.retrieve_tensor(name='episode')

        parameter = tf.train.piecewise_constant(
            x=step, boundaries=self.boundaries, values=self.values
        )

        if util.dtype(x=parameter) != self.dtype:
            parameter = tf.dtypes.cast(x=parameter, dtype=util.tf_dtype(dtype=self.dtype))

        return parameter
