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

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import layer_modules, TensorDict, TensorSpec, TensorsSpec, tf_function, \
    tf_util
from tensorforce.core.distributions import Distribution


class Gaussian(Distribution):
    """
    Gaussian distribution, for continuous actions (specification key: `gaussian`).

    Args:
        global_stddev (bool): Whether to use a separate set of trainable weights to parametrize
            standard deviation, instead of a state-dependent linear transformation
            (<span style="color:#00C000"><b>default</b></span>: false).
        bounded_transform ("clipping" | "tanh"): Transformation to adjust sampled actions in case of
            bounded action space
            (<span style="color:#00C000"><b>default</b></span>: tanh).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, global_stddev=False, bounded_transform='tanh', name=None, action_spec=None,
        input_spec=None
    ):
        assert action_spec.type == 'float'

        parameters_spec = TensorsSpec(
            mean=TensorSpec(type='float', shape=action_spec.shape),
            stddev=TensorSpec(type='float', shape=action_spec.shape),
            log_stddev=TensorSpec(type='float', shape=action_spec.shape)
        )
        conditions_spec = TensorsSpec()

        super().__init__(
            name=name, action_spec=action_spec, input_spec=input_spec,
            parameters_spec=parameters_spec, conditions_spec=conditions_spec
        )

        if global_stddev is None:
            self.global_stddev = False
        else:
            self.global_stddev = global_stddev

        if bounded_transform is None:
            bounded_transform = 'tanh'
        if bounded_transform not in ('clipping', 'tanh'):
            raise TensorforceError.value(
                name='Gaussian', argument='bounded_transform', value=bounded_transform,
                hint='not in {clipping,tanh}'
            )
        elif bounded_transform == 'tanh' and (
            (self.action_spec.min_value is not None) is not (self.action_spec.max_value is not None)
        ):
            raise TensorforceError.value(
                name='Gaussian', argument='bounded_transform', value=bounded_transform,
                condition='one-sided bounded action space'
            )
        elif self.action_spec.min_value is None and self.action_spec.max_value is None:
            bounded_transform = None
        self.bounded_transform = bounded_transform

        if len(self.input_spec.shape) == 1:
            # Single embedding
            action_size = util.product(xs=self.action_spec.shape, empty=0)
            self.mean = self.submodule(
                name='mean', module='linear', modules=layer_modules, size=action_size,
                initialization_scale=0.01, input_spec=self.input_spec
            )
            if not self.global_stddev:
                self.log_stddev = self.submodule(
                    name='log_stddev', module='linear', modules=layer_modules, size=action_size,
                    initialization_scale=0.01, input_spec=self.input_spec
                )

        else:
            # Embedding per action
            if len(self.input_spec.shape) < 1 or len(self.input_spec.shape) > 3:
                raise TensorforceError.value(
                    name=name, argument='input_spec.shape', value=self.embedding_shape,
                    hint='invalid rank'
                )
            elif self.input_spec.shape[:-1] == self.action_spec.shape[:-1]:
                size = self.action_spec.shape[-1]
            elif self.input_spec.shape[:-1] == self.action_spec.shape:
                size = 0
            else:
                raise TensorforceError.value(
                    name=name, argument='input_spec.shape', value=self.input_spec.shape,
                    hint='not flattened and incompatible with action shape'
                )
            self.mean = self.submodule(
                name='mean', module='linear', modules=layer_modules, size=size,
                initialization_scale=0.01, input_spec=self.input_spec
            )
            if not self.global_stddev:
                self.log_stddev = self.submodule(
                    name='log_stddev', module='linear', modules=layer_modules, size=size,
                    initialization_scale=0.01, input_spec=self.input_spec
                )

    def initialize(self):
        super().initialize()

        if self.global_stddev:
            spec = TensorSpec(type='float', shape=((1,) + self.action_spec.shape))
            self.log_stddev = self.variable(
                name='log_stddev', spec=spec, initializer='zeros', is_trainable=True, is_saved=True
            )

        prefix = 'distributions/' + self.name
        self.register_summary(label='distribution', name=(prefix + '-mean', prefix + '-stddev'))

    @tf_function(num_args=2)
    def parametrize(self, *, x, conditions):
        log_epsilon = tf_util.constant(value=np.log(util.epsilon), dtype='float')
        shape = (-1,) + self.action_spec.shape

        # Mean
        mean = self.mean.apply(x=x)
        if len(self.input_spec.shape) == 1:
            mean = tf.reshape(tensor=mean, shape=shape)

        # Log standard deviation
        if self.global_stddev:
            multiples = (tf.shape(input=x)[0],) + tuple(1 for _ in range(self.action_spec.rank))
            log_stddev = tf.tile(input=self.log_stddev, multiples=multiples)
        else:
            log_stddev = self.log_stddev.apply(x=x)
            if len(self.input_spec.shape) == 1:
                log_stddev = tf.reshape(tensor=log_stddev, shape=shape)

        # Shift log stddev to reduce zero value (TODO: 0.1 random choice)
        if self.action_spec.min_value is not None and self.action_spec.max_value is not None:
            log_stddev += tf_util.constant(value=np.log(0.1), dtype='float')

        # Clip log_stddev for numerical stability (epsilon < 1.0, hence negative)
        log_stddev = tf.clip_by_value(
            t=log_stddev, clip_value_min=log_epsilon, clip_value_max=-log_epsilon
        )

        # Standard deviation
        stddev = tf.math.exp(x=log_stddev)

        return TensorDict(mean=mean, stddev=stddev, log_stddev=log_stddev)

    @tf_function(num_args=1)
    def mode(self, *, parameters):
        action = parameters['mean']

        # Bounded transformation
        if self.bounded_transform is not None:
            if self.bounded_transform == 'tanh':
                action = tf.math.tanh(x=action)

            if self.action_spec.min_value is not None and self.action_spec.max_value is not None:
                one = tf_util.constant(value=1.0, dtype='float')
                half = tf_util.constant(value=0.5, dtype='float')
                min_value = tf_util.constant(value=self.action_spec.min_value, dtype='float')
                max_value = tf_util.constant(value=self.action_spec.max_value, dtype='float')
                action = min_value + (max_value - min_value) * half * (action + one)

            elif self.action_spec.min_value is not None:
                min_value = tf_util.constant(value=self.action_spec.min_value, dtype='float')
                action = tf.maximum(x=min_value, y=action)
            else:
                assert self.action_spec.max_value is not None
                max_value = tf_util.constant(value=self.action_spec.max_value, dtype='float')
                action = tf.minimum(x=max_value, y=action)

        return action

    @tf_function(num_args=2)
    def sample(self, *, parameters, temperature):
        mean, stddev, log_stddev = parameters.get(('mean', 'stddev', 'log_stddev'))

        # Distribution parameter summaries
        def fn_summary():
            return tf.math.reduce_mean(input_tensor=mean, axis=range(self.action_spec.rank + 1)), \
                tf.math.reduce_mean(input_tensor=stddev, axis=range(self.action_spec.rank + 1))

        prefix = 'distributions/' + self.name
        dependencies = self.summary(
            label='distribution', name=(prefix + '-mean', prefix + '-stddev'), data=fn_summary,
            step='timesteps'
        )

        def fn_mode():
            return mean

        def fn_sample():
            normal_distribution = tf.random.normal(
                shape=tf.shape(input=mean), dtype=tf_util.get_dtype(type='float')
            )
            return mean + stddev * temperature * normal_distribution

        with tf.control_dependencies(control_inputs=dependencies):
            epsilon = tf_util.constant(value=util.epsilon, dtype='float')
            action = tf.cond(pred=(temperature < epsilon), true_fn=fn_mode, false_fn=fn_sample)

            # Bounded transformation
            if self.bounded_transform is not None:
                if self.bounded_transform == 'tanh':
                    action = tf.math.tanh(x=action)

                if self.action_spec.min_value is not None and \
                        self.action_spec.max_value is not None:
                    one = tf_util.constant(value=1.0, dtype='float')
                    half = tf_util.constant(value=0.5, dtype='float')
                    min_value = tf_util.constant(value=self.action_spec.min_value, dtype='float')
                    max_value = tf_util.constant(value=self.action_spec.max_value, dtype='float')
                    action = min_value + (max_value - min_value) * half * (action + one)

                elif self.action_spec.min_value is not None:
                    min_value = tf_util.constant(value=self.action_spec.min_value, dtype='float')
                    action = tf.maximum(x=min_value, y=action)
                else:
                    assert self.action_spec.max_value is not None
                    max_value = tf_util.constant(value=self.action_spec.max_value, dtype='float')
                    action = tf.minimum(x=max_value, y=action)

            return action

    @tf_function(num_args=2)
    def log_probability(self, *, parameters, action):
        mean, stddev, log_stddev = parameters.get(('mean', 'stddev', 'log_stddev'))

        # Inverse bounded transformation
        if self.bounded_transform is not None:
            if self.action_spec.min_value is not None and self.action_spec.max_value is not None:
                one = tf_util.constant(value=1.0, dtype='float')
                two = tf_util.constant(value=2.0, dtype='float')
                min_value = tf_util.constant(value=self.action_spec.min_value, dtype='float')
                max_value = tf_util.constant(value=self.action_spec.max_value, dtype='float')
                action = two * (action - min_value) / (max_value - min_value) - one

            if self.bounded_transform == 'tanh':
                clip = tf_util.constant(value=(1.0 - util.epsilon), dtype='float')
                action = tf.clip_by_value(t=action, clip_value_min=-clip, clip_value_max=clip)
                action = tf.math.atanh(x=action)

        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        half = tf_util.constant(value=0.5, dtype='float')
        half_log_two_pi = tf_util.constant(value=(0.5 * np.log(2.0 * np.pi)), dtype='float')

        sq_mean_distance = tf.square(x=(action - mean))
        sq_stddev = tf.maximum(x=tf.square(x=stddev), y=epsilon)

        log_prob = -half * sq_mean_distance / sq_stddev - log_stddev - half_log_two_pi

        if self.bounded_transform == 'tanh':
            log_two = tf_util.constant(value=np.log(2.0), dtype='float')
            log_prob -= two * (log_two - action - tf.math.softplus(features=(-two * action)))

        return log_prob

    @tf_function(num_args=1)
    def entropy(self, *, parameters):
        log_stddev = parameters['log_stddev']

        half_lg_two_pi_e = tf_util.constant(value=(0.5 * np.log(2.0 * np.pi * np.e)), dtype='float')

        # TODO: doesn't take into account self.bounded_transform == 'tanh'

        return log_stddev + half_lg_two_pi_e

    @tf_function(num_args=2)
    def kl_divergence(self, *, parameters1, parameters2):
        mean1, stddev1, log_stddev1 = parameters1.get(('mean', 'stddev', 'log_stddev'))
        mean2, stddev2, log_stddev2 = parameters2.get(('mean', 'stddev', 'log_stddev'))

        half = tf_util.constant(value=0.5, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        log_stddev_ratio = log_stddev2 - log_stddev1
        sq_mean_distance = tf.square(x=(mean1 - mean2))
        sq_stddev1 = tf.square(x=stddev1)
        sq_stddev2 = tf.maximum(x=tf.square(x=stddev2), y=epsilon)

        return log_stddev_ratio + half * (sq_stddev1 + sq_mean_distance) / sq_stddev2 - half

    @tf_function(num_args=2)
    def action_value(self, *, parameters, action):
        mean, stddev, log_stddev = parameters.get(('mean', 'stddev', 'log_stddev'))

        # Inverse bounded transformation
        if self.bounded_transform is not None:
            if self.action_spec.min_value is not None and self.action_spec.max_value is not None:
                one = tf_util.constant(value=1.0, dtype='float')
                two = tf_util.constant(value=2.0, dtype='float')
                min_value = tf_util.constant(value=self.action_spec.min_value, dtype='float')
                max_value = tf_util.constant(value=self.action_spec.max_value, dtype='float')
                action = two * (action - min_value) / (max_value - min_value) - one

            if self.bounded_transform == 'tanh':
                clip = tf_util.constant(value=(1.0 - util.epsilon), dtype='float')
                action = tf.clip_by_value(t=action, clip_value_min=-clip, clip_value_max=clip)
                action = tf.math.atanh(x=action)

        half = tf_util.constant(value=0.5, dtype='float')
        two = tf_util.constant(value=2.0, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        log_two_pi = tf_util.constant(value=(np.log(2.0 * np.pi)), dtype='float')
        # TODO: why no e here, but for entropy?

        sq_mean_distance = tf.square(x=(action - mean))
        sq_stddev = tf.maximum(x=tf.square(x=stddev), y=epsilon)

        action_value = -half * sq_mean_distance / sq_stddev - two * log_stddev - log_two_pi

        # Probably not needed?
        # if self.bounded_transform == 'tanh':
        #     log_two = tf_util.constant(value=np.log(2.0), dtype='float')
        #     action_value -= two * (log_two - action - tf.math.softplus(features=(-two * action)))

        return action_value

    @tf_function(num_args=1)
    def state_value(self, *, parameters):
        log_stddev = parameters['log_stddev']

        half_lg_two_pi = tf_util.constant(value=(0.5 * np.log(2.0 * np.pi)), dtype='float')
        # TODO: why no e here, but for entropy?

        return -log_stddev - half_lg_two_pi
