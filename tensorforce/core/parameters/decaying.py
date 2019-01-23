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


class Decaying(Parameter):
    """
    Decaying hyperparameter.
    """

    def __init__(
        self, name, dtype, unit, decay, initial_value, decay_steps, inverse=False, scale=1.0,
        summary_labels=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, summary_labels=summary_labels)

        assert unit in ('timesteps', 'episodes')
        assert decay in (
            'cosine', 'cosine_restarts', 'exponential', 'inverse_time', 'linear_cosine',
            'linear_cosine_noisy', 'natural_exponential', 'polynomial'
        )
        assert isinstance(initial_value, float)
        assert isinstance(decay_steps, int)

        self.unit = unit
        self.decay = decay
        self.initial_value = initial_value
        self.decay_steps = decay_steps
        self.inverse = inverse
        self.scale = scale
        self.kwargs = kwargs

    def get_parameter_value(self):
        if self.unit == 'timesteps':
            step = Module.retrieve_tensor(name='timestep')
        elif self.unit == 'episodes':
            step = Module.retrieve_tensor(name='episode')

        initial_value = tf.constant(value=self.initial_value, dtype=util.tf_dtype(dtype='float'))

        if self.decay == 'cosine':
            parameter = tf.train.cosine_decay(
                learning_rate=initial_value, global_step=step, decay_steps=self.decay_steps,
                alpha=self.kwargs.get('alpha', 0.0)
            )

        elif self.decay == 'cosine_restarts':
            parameter = tf.train.cosine_decay_restarts(
                learning_rate=initial_value, global_step=step,
                first_decay_steps=self.decay_steps, t_mul=self.kwargs.get('t_mul', 2.0),
                m_mul=self.kwargs.get('m_mul', 1.0), alpha=self.kwargs.get('alpha', 0.0)
            )

        elif self.decay == 'exponential':
            parameter = tf.train.exponential_decay(
                learning_rate=initial_value, global_step=step, decay_steps=self.decay_steps,
                decay_rate=self.kwargs['decay_rate'], staircase=self.kwargs.get('staircase', False)
            )

        elif self.decay == 'inverse_time':
            parameter = tf.train.inverse_time_decay(
                learning_rate=initial_value, global_step=step, decay_steps=self.decay_steps,
                decay_rate=self.kwargs['decay_rate'], staircase=self.kwargs.get('staircase', False)
            )

        elif self.decay == 'linear_cosine':
            parameter = tf.train.linear_cosine_decay(
                learning_rate=initial_value, global_step=step, decay_steps=self.decay_steps,
                num_periods=self.kwargs.get('num_periods', 0.5),
                alpha=self.kwargs.get('alpha', 0.0), beta=self.kwargs.get('beta', 0.001)
            )

        elif self.decay == 'linear_cosine_noisy':
            parameter = tf.train.noisy_linear_cosine_decay(
                learning_rate=initial_value, global_step=step, decay_steps=self.decay_steps,
                initial_variance=self.kwargs.get('initial_variance', 1.0),
                variance_decay=self.kwargs.get('variance_decay', 0.55),
                num_periods=self.kwargs.get('num_periods', 0.5),
                alpha=self.kwargs.get('alpha', 0.0), beta=self.kwargs.get('beta', 0.001)
            )
        elif self.decay == 'natural_exponential':
            parameter = tf.train.natural_exp_decay(
                learning_rate=initial_value, global_step=step, decay_steps=self.decay_steps,
                decay_rate=self.kwargs['decay_rate'], staircase=self.kwargs.get('staircase', False)
            )

        elif self.decay == 'polynomial':
            parameter = tf.train.polynomial_decay(
                learning_rate=initial_value, global_step=step, decay_steps=self.decay_steps,
                end_learning_rate=self.kwargs.get('final_value', 0.0001),
                power=self.kwargs.get('power', 1.0), cycle=self.kwargs.get('cycle', False)
            )

        if self.decay:
            one = tf.constant(value=1.0, dtype=util.tf_dtype(dtype='float'))
            parameter = one - parameter

        if self.scale != 1.0:
            scale = tf.constant(value=self.scale, dtype=util.tf_dtype(dtype='float'))
            parameter = parameter * scale

        if self.dtype != 'float':
            parameter = tf.dtypes.cast(x=parameter, dtype=util.tf_dtype(dtype=self.dtype))

        return parameter
