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

from tensorforce.core import TensorSpec, tf_util
from tensorforce.core.parameters import Parameter


class OrnsteinUhlenbeck(Parameter):
    """
    Ornstein-Uhlenbeck process (specification key: `ornstein_uhlenbeck`).

    Args:
        theta (float > 0.0): Theta value
            (<span style="color:#00C000"><b>default</b></span>: 0.15).
        sigma (float > 0.0): Sigma value
            (<span style="color:#00C000"><b>default</b></span>: 0.3).
        mu (float): Mu value
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        absolute (bool): Absolute value
            (<span style="color:#00C000"><b>default</b></span>: false).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        dtype (type): <span style="color:#0000C0"><b>internal use</b></span>.
        min_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
        max_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, theta=0.15, sigma=0.3, mu=0.0, absolute=False, name=None, dtype=None,
        min_value=None, max_value=None
    ):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.absolute = absolute

        super().__init__(name=name, dtype=dtype, min_value=min_value, max_value=max_value)

    def min_value(self):
        if self.absolute:
            return self.spec.py_type()(0.0)
        else:
            super().min_value()

    def final_value(self):
        return self.spec.py_type()(self.mu)

    def initialize(self):
        super().initialize()

        self.process = self.variable(
            name='process', spec=TensorSpec(type='float'), initializer=self.mu, is_trainable=False,
            is_saved=True
        )

    def parameter_value(self, *, step):
        delta = self.theta * (self.mu - self.process) + self.sigma * tf.random.normal(shape=())
        if self.absolute:
            parameter = self.process.assign(value=tf.math.abs(x=(self.process + delta)))
        else:
            parameter = self.process.assign_add(delta=delta)

        parameter = tf_util.cast(x=parameter, dtype=self.spec.type)

        return parameter
