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

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import layer_modules, TensorDict, TensorSpec, TensorsSpec, tf_function, \
    tf_util
from tensorforce.core.distributions import Distribution


class Gaussian(Distribution):
    """
    Gaussian distribution, for unbounded continuous actions (specification key: `gaussian`).

    Args:
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, summary_labels=None, name=None, action_spec=None, input_spec=None):
        parameters_spec = TensorsSpec(
            mean=TensorSpec(type='float', shape=action_spec.shape),
            stddev=TensorSpec(type='float', shape=action_spec.shape),
            log_stddev=TensorSpec(type='float', shape=action_spec.shape)
        )

        super().__init__(
            summary_labels=summary_labels, name=name, action_spec=action_spec,
            input_spec=input_spec, parameters_spec=parameters_spec
        )

        if len(self.input_spec.shape) == 1:
            action_size = util.product(xs=self.action_spec.shape, empty=0)
            self.mean = self.add_module(
                name='mean', module='linear', modules=layer_modules, size=action_size,
                input_spec=self.input_spec
            )
            self.log_stddev = self.add_module(
                name='log_stddev', module='linear', modules=layer_modules, size=action_size,
                input_spec=self.input_spec
            )

        else:
            if len(self.input_spec.shape) < 1 or len(self.input_spec.shape) > 3:
                raise TensorforceError.value(
                    name=name, argument='input_spec.shape', value=self.embedding_shape,
                    hint='invalid rank'
                )
            if self.input_spec.shape[:-1] == self.action_spec.shape[:-1]:
                size = self.action_spec.shape[-1]
            elif self.input_spec.shape[:-1] == self.action_spec.shape:
                size = 0
            else:
                raise TensorforceError.value(
                    name=name, argument='input_spec.shape', value=self.input_spec.shape,
                    hint='not flattened and incompatible with action shape'
                )
            self.mean = self.add_module(
                name='mean', module='linear', modules=layer_modules, size=size,
                input_spec=self.input_spec
            )
            self.log_stddev = self.add_module(
                name='log_stddev', module='linear', modules=layer_modules, size=size,
                input_spec=self.input_spec
            )

    @tf_function(num_args=1)
    def parametrize(self, x):
        log_epsilon = tf_util.constant(value=np.log(util.epsilon), dtype='float')
        shape = (-1,) + self.action_spec.shape

        # Mean
        mean = self.mean.apply(x=x)
        if len(self.input_spec.shape) == 1:
            mean = tf.reshape(tensor=mean, shape=shape)

        # Log standard deviation
        log_stddev = self.log_stddev.apply(x=x)
        if len(self.input_spec.shape) == 1:
            log_stddev = tf.reshape(tensor=log_stddev, shape=shape)

        # Clip log_stddev for numerical stability (epsilon < 1.0, hence negative)
        log_stddev = tf.clip_by_value(
            t=log_stddev, clip_value_min=log_epsilon, clip_value_max=-log_epsilon
        )

        # Standard deviation
        stddev = tf.exp(x=log_stddev)

        return TensorDict(mean=mean, stddev=stddev, log_stddev=log_stddev)

    @tf_function(num_args=2)
    def sample(self, parameters, temperature):
        mean, stddev = parameters.get('mean', 'stddev')

        summary_mean = mean
        summary_stddev = stddev
        for _ in range(len(self.action_spec.shape)):
            summary_mean = tf.math.reduce_mean(input_tensor=summary_mean, axis=1)
            summary_stddev = tf.math.reduce_mean(input_tensor=summary_stddev, axis=1)

        mean, stddev = self.add_summary(
            label=('distributions', 'gaussian'), name='mean', tensor=summary_mean,
            pass_tensors=(mean, stddev)
        )
        mean, stddev = self.add_summary(
            label=('distributions', 'gaussian'), name='stddev', tensor=summary_stddev,
            pass_tensors=(mean, stddev)
        )

        normal_distribution = tf.random.normal(
            shape=tf.shape(input=mean), dtype=tf_util.get_dtype(type='float')
        )
        action = mean + stddev * temperature * normal_distribution

        # Clip if bounded action
        if self.action_spec.min_value is not None:
            action = tf.maximum(x=self.action_spec.min_value, y=action)
        if self.action_spec.max_value is not None:
            action = tf.minimum(x=self.action_spec.max_value, y=action)

        return action

    @tf_function(num_args=2)
    def log_probability(self, parameters, action):
        mean, stddev, log_stddev = parameters.get('mean', 'stddev', 'log_stddev')

        half = tf_util.constant(value=0.5, dtype='float')
        two = tf_util.constant(value=2.0, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        pi_const = tf_util.constant(value=np.pi, dtype='float')

        sq_mean_distance = tf.square(x=(action - mean))
        sq_stddev = tf.maximum(x=tf.square(x=stddev), y=epsilon)

        return -half * sq_mean_distance / sq_stddev - log_stddev - \
            half * tf.math.log(x=(two * pi_const))

    @tf_function(num_args=1)
    def entropy(self, parameters):
        log_stddev = parameters['log_stddev']

        half = tf_util.constant(value=0.5, dtype='float')
        two = tf_util.constant(value=2.0, dtype='float')
        e_const = tf_util.constant(value=np.e, dtype='float')
        pi_const = tf_util.constant(value=np.pi, dtype='float')

        return log_stddev + half * tf.math.log(x=(two * pi_const * e_const))

    @tf_function(num_args=2)
    def kl_divergence(self, parameters1, parameters2):
        mean1, stddev1, log_stddev1 = parameters1.get('mean', 'stddev', 'log_stddev')
        mean2, stddev2, log_stddev2 = parameters2.get('mean', 'stddev', 'log_stddev')

        half = tf_util.constant(value=0.5, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        log_stddev_ratio = log_stddev2 - log_stddev1
        sq_mean_distance = tf.square(x=(mean1 - mean2))
        sq_stddev1 = tf.square(x=stddev1)
        sq_stddev2 = tf.maximum(x=tf.square(x=stddev2), y=epsilon)

        return log_stddev_ratio + half * (sq_stddev1 + sq_mean_distance) / sq_stddev2 - half

    @tf_function(num_args=2)
    def action_value(self, parameters, action):
        mean, stddev, log_stddev = parameters.get('mean', 'stddev', 'log_stddev')

        half = tf_util.constant(value=0.5, dtype='float')
        two = tf_util.constant(value=2.0, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        pi_const = tf_util.constant(value=np.pi, dtype='float')

        sq_mean_distance = tf.square(x=(action - mean))
        sq_stddev = tf.maximum(x=tf.square(x=stddev), y=epsilon)

        return -half * sq_mean_distance / sq_stddev - two * log_stddev - \
            tf.math.log(x=(two * pi_const))

    @tf_function(num_args=1)
    def states_value(self, parameters):
        log_stddev = parameters['log_stddev']

        half = tf_util.constant(value=0.5, dtype='float')
        two = tf_util.constant(value=2.0, dtype='float')
        pi_const = tf_util.constant(value=np.pi, dtype='float')

        return -log_stddev - half * tf.math.log(x=(two * pi_const))
