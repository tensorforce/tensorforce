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
        stddev_mode ("predicted" | "global"): Whether to predict the standard deviation via a linear
            transformation of the state embedding, or to parametrize the standard deviation by a
            separate set of trainable weights
            (<span style="color:#00C000"><b>default</b></span>: "predicted").
        bounded_transform ("clipping" | "tanh"): Transformation to adjust sampled actions in case of
            bounded action space, "tanh" transforms distribution (e.g. log probability computation)
            accordingly whereas "clipping" does not
            (<span style="color:#00C000"><b>default</b></span>: tanh).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, stddev_mode='predicted', bounded_transform='tanh', name=None, action_spec=None,
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

        self.stddev_mode = stddev_mode

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

        if self.input_spec.rank == 1:
            # Single embedding
            self.mean = self.submodule(
                name='mean', module='linear', modules=layer_modules, size=self.action_spec.size,
                initialization_scale=0.01, input_spec=self.input_spec
            )
            if self.stddev_mode == 'predicted':
                self.stddev = self.submodule(
                    name='stddev', module='linear', modules=layer_modules,
                    size=self.action_spec.size, initialization_scale=0.01,
                    input_spec=self.input_spec
                )

        else:
            # Embedding per action
            if self.input_spec.rank < 1 or self.input_spec.rank > 3:
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
            if self.stddev_mode == 'predicted':
                self.stddev = self.submodule(
                    name='stddev', module='linear', modules=layer_modules, size=size,
                    initialization_scale=0.01, input_spec=self.input_spec
                )

    def get_architecture(self):
        architecture = 'Mean:  {}'.format(self.mean.get_architecture())
        if self.stddev_mode == 'predicted':
            architecture += '\nStddev:  {}'.format(self.stddev.get_architecture())
        return architecture

    def initialize(self):
        super().initialize()

        if self.stddev_mode == 'global':
            spec = TensorSpec(type='float', shape=((1,) + self.action_spec.shape))
            self.stddev = self.variable(
                name='stddev', spec=spec, initializer='zeros', is_trainable=True, is_saved=True
            )

        prefix = 'distributions/' + self.name
        names = (prefix + '-mean', prefix + '-stddev')
        self.register_summary(label='distribution', name=names)

        spec = self.parameters_spec['mean']
        self.register_tracking(label='distribution', name='mean', spec=spec)
        self.register_tracking(label='distribution', name='stddev', spec=spec)

    @tf_function(num_args=2)
    def parametrize(self, *, x, conditions):
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        log_epsilon = tf_util.constant(value=np.log(util.epsilon), dtype='float')

        # Mean
        mean = self.mean.apply(x=x)
        if self.input_spec.rank == 1:
            shape = (-1,) + self.action_spec.shape
            mean = tf.reshape(tensor=mean, shape=shape)

        # Softplus standard deviation
        if self.stddev_mode == 'global':
            multiples = (tf.shape(input=x)[0],) + tuple(1 for _ in range(self.action_spec.rank))
            softplus_stddev = tf.tile(input=self.stddev, multiples=multiples)
        else:
            softplus_stddev = self.stddev.apply(x=x)
            if self.input_spec.rank == 1:
                softplus_stddev = tf.reshape(tensor=softplus_stddev, shape=shape)

        # # Shift softplus_stddev to reduce zero value to 0.25 (TODO: 0.25 random choice)
        # if self.action_spec.min_value is not None and self.action_spec.max_value is not None:
        #     softplus_stddev += tf_util.constant(value=np.log(0.25), dtype='float')

        # Clip softplus_stddev for numerical stability (epsilon < 1.0, hence negative)
        softplus_stddev = tf.clip_by_value(
            t=softplus_stddev, clip_value_min=log_epsilon, clip_value_max=-log_epsilon
        )

        # Softplus transformation (based on https://arxiv.org/abs/2007.06059)
        softplus_shift = tf_util.constant(value=0.2, dtype='float')
        log_two = tf_util.constant(value=np.log(2.0), dtype='float')
        stddev = (tf.nn.softplus(features=softplus_stddev) + softplus_shift) / \
            (log_two + softplus_shift)

        # Divide stddev to reduce zero value to 0.25 (TODO: 0.25 random choice)
        if self.action_spec.min_value is not None and self.action_spec.max_value is not None:
            stddev *= tf_util.constant(value=0.25, dtype='float')

        # Log stddev
        log_stddev = tf.math.log(x=(stddev + epsilon))

        return TensorDict(mean=mean, stddev=stddev, log_stddev=log_stddev)

    @tf_function(num_args=1)
    def mode(self, *, parameters, independent):
        mean, stddev = parameters.get(('mean', 'stddev'))

        # Distribution parameter summaries and tracking
        dependencies = list()
        if not independent:
            def fn_summary():
                m = tf.math.reduce_mean(input_tensor=mean, axis=range(self.action_spec.rank + 1))
                s = tf.math.reduce_mean(input_tensor=stddev, axis=range(self.action_spec.rank + 1))
                return m, s

            prefix = 'distributions/' + self.name
            names = (prefix + '-mean', prefix + '-stddev')
            dependencies.extend(self.summary(
                label='distribution', name=names, data=fn_summary, step='timesteps'
            ))

        # Distribution parameter tracking
        def fn_tracking():
            return tf.math.reduce_mean(input_tensor=mean, axis=0)

        dependencies.extend(self.track(label='distribution', name='mean', data=fn_tracking))

        def fn_tracking():
            return tf.math.reduce_mean(input_tensor=stddev, axis=0)

        dependencies.extend(self.track(label='distribution', name='stddev', data=fn_tracking))

        with tf.control_dependencies(control_inputs=dependencies):
            action = mean

            # Bounded transformation
            if self.bounded_transform is not None:
                one = tf_util.constant(value=1.0, dtype='float')

                if self.bounded_transform == 'tanh':
                    action = tf.math.tanh(x=action)
                elif self.bounded_transform == 'clipping':
                    action = tf.clip_by_value(t=action, clip_value_min=-one, clip_value_max=one)

                if self.action_spec.min_value is not None and \
                        self.action_spec.max_value is not None:
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
    def sample(self, *, parameters, temperature, independent):
        mean, stddev = parameters.get(('mean', 'stddev'))

        # Distribution parameter summaries and tracking
        dependencies = list()
        if not independent:
            def fn_summary():
                m = tf.math.reduce_mean(input_tensor=mean, axis=range(self.action_spec.rank + 1))
                s = tf.math.reduce_mean(input_tensor=stddev, axis=range(self.action_spec.rank + 1))
                return m, s

            prefix = 'distributions/' + self.name
            names = (prefix + '-mean', prefix + '-stddev')
            dependencies.extend(self.summary(
                label='distribution', name=names, data=fn_summary, step='timesteps'
            ))

        # Distribution parameter tracking
        def fn_tracking():
            return tf.math.reduce_mean(input_tensor=mean, axis=0)

        dependencies.extend(self.track(label='distribution', name='mean', data=fn_tracking))

        def fn_tracking():
            return tf.math.reduce_mean(input_tensor=stddev, axis=0)

        dependencies.extend(self.track(label='distribution', name='stddev', data=fn_tracking))

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
                one = tf_util.constant(value=1.0, dtype='float')

                if self.bounded_transform == 'tanh':
                    action = tf.math.tanh(x=action)
                elif self.bounded_transform == 'clipping':
                    action = tf.clip_by_value(t=action, clip_value_min=-one, clip_value_max=one)

                if self.action_spec.min_value is not None and \
                        self.action_spec.max_value is not None:
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
                action = tf_util.cast(x=tf.math.atanh(x=tf_util.float32(x=action)), dtype='float')

        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        half = tf_util.constant(value=0.5, dtype='float')
        half_log_two_pi = tf_util.constant(value=(0.5 * np.log(2.0 * np.pi)), dtype='float')

        sq_mean_distance = tf.square(x=(action - mean))
        sq_stddev = tf.square(x=stddev) + epsilon

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
        sq_stddev2 = tf.square(x=stddev2) + epsilon

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
        sq_stddev = tf.square(x=stddev) + epsilon

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
