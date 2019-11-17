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

from math import e, log, pi

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import layer_modules, Module
from tensorforce.core.distributions import Distribution


class Gaussian(Distribution):
    """
    Gaussian distribution, for unbounded continuous actions (specification key: `gaussian`).

    Args:
        name (string): Distribution name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        action_spec (specification): Action specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        embedding_shape (iter[int > 0]): Embedding shape
            (<span style="color:#0000C0"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, action_spec, embedding_shape, summary_labels=None):
        super().__init__(
            name=name, action_spec=action_spec, embedding_shape=embedding_shape,
            summary_labels=summary_labels
        )

        input_spec = dict(type='float', shape=self.embedding_shape)

        if len(self.embedding_shape) == 1:
            action_size = util.product(xs=self.action_spec['shape'], empty=0)
            self.mean = self.add_module(
                name='mean', module='linear', modules=layer_modules, size=action_size,
                input_spec=input_spec
            )
            self.log_stddev = self.add_module(
                name='log-stddev', module='linear', modules=layer_modules, size=action_size,
                input_spec=input_spec
            )

        else:
            if len(self.embedding_shape) < 1 or len(self.embedding_shape) > 3:
                raise TensorforceError.unexpected()
            if self.embedding_shape[:-1] == self.action_spec['shape'][:-1]:
                size = self.action_spec['shape'][-1]
            elif self.embedding_shape[:-1] == self.action_spec['shape']:
                size = 0
            else:
                raise TensorforceError.unexpected()
            self.mean = self.add_module(
                name='mean', module='linear', modules=layer_modules, size=size,
                input_spec=input_spec
            )
            self.log_stddev = self.add_module(
                name='log-stddev', module='linear', modules=layer_modules, size=size,
                input_spec=input_spec
            )

        Module.register_tensor(
            name=(self.name + '-mean'), spec=dict(type='float', shape=self.action_spec['shape']),
            batched=True
        )
        Module.register_tensor(
            name=(self.name + '-stddev'), spec=dict(type='float', shape=self.action_spec['shape']),
            batched=True
        )

    def tf_parametrize(self, x):
        log_epsilon = tf.constant(value=log(util.epsilon), dtype=util.tf_dtype(dtype='float'))
        shape = (-1,) + self.action_spec['shape']

        # Mean
        mean = self.mean.apply(x=x)
        if len(self.embedding_shape) == 1:
            mean = tf.reshape(tensor=mean, shape=shape)

        # Log standard deviation
        log_stddev = self.log_stddev.apply(x=x)
        if len(self.embedding_shape) == 1:
            log_stddev = tf.reshape(tensor=log_stddev, shape=shape)

        # Clip log_stddev for numerical stability
        # epsilon < 1.0, hence negative
        log_stddev = tf.clip_by_value(
            t=log_stddev, clip_value_min=log_epsilon, clip_value_max=-log_epsilon
        )

        # Standard deviation
        stddev = tf.exp(x=log_stddev)

        Module.update_tensor(name=(self.name + '-mean'), tensor=mean)
        Module.update_tensor(name=(self.name + '-stddev'), tensor=stddev)

        return mean, stddev, log_stddev

    def tf_sample(self, parameters, temperature):
        mean, stddev, _ = parameters

        summary_mean = mean
        summary_stddev = stddev
        for _ in range(len(self.action_spec['shape'])):
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
            shape=tf.shape(input=mean), dtype=util.tf_dtype(dtype='float')
        )
        action = mean + stddev * temperature * normal_distribution

        # Clip if bounded action
        if 'min_value' in self.action_spec:
            min_value = tf.constant(
                value=self.action_spec['min_value'], dtype=util.tf_dtype(dtype='float')
            )
            max_value = tf.constant(
                value=self.action_spec['max_value'], dtype=util.tf_dtype(dtype='float')
            )
            action = tf.clip_by_value(t=action, clip_value_min=min_value, clip_value_max=max_value)

        return action

    def tf_log_probability(self, parameters, action):
        mean, stddev, log_stddev = parameters

        half = tf.constant(value=0.5, dtype=util.tf_dtype(dtype='float'))
        two = tf.constant(value=2.0, dtype=util.tf_dtype(dtype='float'))
        epsilon = tf.constant(value=util.epsilon, dtype=util.tf_dtype(dtype='float'))
        pi_const = tf.constant(value=pi, dtype=util.tf_dtype(dtype='float'))

        sq_mean_distance = tf.square(x=(action - mean))
        sq_stddev = tf.maximum(x=tf.square(x=stddev), y=epsilon)

        return -half * sq_mean_distance / sq_stddev - log_stddev - \
            half * tf.math.log(x=(two * pi_const))

    def tf_entropy(self, parameters):
        _, _, log_stddev = parameters

        half = tf.constant(value=0.5, dtype=util.tf_dtype(dtype='float'))
        two = tf.constant(value=2.0, dtype=util.tf_dtype(dtype='float'))
        e_const = tf.constant(value=e, dtype=util.tf_dtype(dtype='float'))
        pi_const = tf.constant(value=pi, dtype=util.tf_dtype(dtype='float'))

        return log_stddev + half * tf.math.log(x=(two * pi_const * e_const))

    def tf_kl_divergence(self, parameters1, parameters2):
        mean1, stddev1, log_stddev1 = parameters1
        mean2, stddev2, log_stddev2 = parameters2

        half = tf.constant(value=0.5, dtype=util.tf_dtype(dtype='float'))
        epsilon = tf.constant(value=util.epsilon, dtype=util.tf_dtype(dtype='float'))

        log_stddev_ratio = log_stddev2 - log_stddev1
        sq_mean_distance = tf.square(x=(mean1 - mean2))
        sq_stddev1 = tf.square(x=stddev1)
        sq_stddev2 = tf.maximum(x=tf.square(x=stddev2), y=epsilon)

        return log_stddev_ratio + half * (sq_stddev1 + sq_mean_distance) / sq_stddev2 - half

    def tf_action_value(self, parameters, action):
        mean, stddev, log_stddev = parameters

        half = tf.constant(value=0.5, dtype=util.tf_dtype(dtype='float'))
        two = tf.constant(value=2.0, dtype=util.tf_dtype(dtype='float'))
        epsilon = tf.constant(value=util.epsilon, dtype=util.tf_dtype(dtype='float'))
        pi_const = tf.constant(value=pi, dtype=util.tf_dtype(dtype='float'))

        sq_mean_distance = tf.square(x=(action - mean))
        sq_stddev = tf.maximum(x=tf.square(x=stddev), y=epsilon)

        return -half * sq_mean_distance / sq_stddev - two * log_stddev - \
            tf.math.log(x=(two * pi_const))

    def tf_states_value(self, parameters):
        _, _, log_stddev = parameters

        half = tf.constant(value=0.5, dtype=util.tf_dtype(dtype='float'))
        two = tf.constant(value=2.0, dtype=util.tf_dtype(dtype='float'))
        pi_const = tf.constant(value=pi, dtype=util.tf_dtype(dtype='float'))

        return -log_stddev - half * tf.math.log(x=(two * pi_const))
