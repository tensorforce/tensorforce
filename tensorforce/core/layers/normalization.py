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

from tensorforce import TensorforceError
from tensorforce.core import parameter_modules, TensorSpec, tf_function, tf_util
from tensorforce.core.layers import Layer, StatefulLayer


class LinearNormalization(Layer):
    """
    Linear normalization layer which scales and shifts the input to [-2.0, 2.0], for bounded states
    with min/max_value (specification key: `linear_normalization`).

    Args:
        min_value (float | array[float]): Lower bound of the value
            (<span style="color:#00C000"><b>default</b></span>: based on input_spec).
        max_value (float | array[float]): Upper bound of the value range
            (<span style="color:#00C000"><b>default</b></span>: based on input_spec).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, min_value=None, max_value=None, name=None, input_spec=None):
        if min_value is None:
            if input_spec.min_value is None:
                raise TensorforceError.required(name='LinearNormalization', argument='min_value')
            min_value = input_spec.min_value

        if max_value is None:
            if input_spec.max_value is None:
                raise TensorforceError.required(name='LinearNormalization', argument='max_value')
            max_value = input_spec.max_value

        self.min_value = np.asarray(min_value)
        self.max_value = np.asarray(max_value)

        if (self.min_value >= self.max_value).any():
            raise TensorforceError(
                name='LinearNormalization', argument='min/max_value',
                value=(self.min_value, self.max_value), hint='not less than'
            )

        super().__init__(name=name, input_spec=input_spec)

    def default_input_spec(self):
        return TensorSpec(
            type='float', shape=None, min_value=self.min_value, max_value=self.max_value
        )

    def output_spec(self):
        output_spec = super().output_spec()
        is_inf = np.logical_or(np.isinf(self.min_value), np.isinf(self.max_value))
        if is_inf.any():
            output_spec.min_value = np.where(is_inf, self.min_value, -2.0)
            output_spec.max_value = np.where(is_inf, self.max_value, 2.0)
        else:
            output_spec.min_value = -2.0
            output_spec.max_value = 2.0
        return output_spec

    @tf_function(num_args=1)
    def apply(self, *, x):
        is_inf = np.logical_or(np.isinf(self.min_value), np.isinf(self.max_value))
        is_inf = tf_util.constant(value=is_inf, dtype='bool')
        min_value = tf_util.constant(value=self.min_value, dtype='float')
        max_value = tf_util.constant(value=self.max_value, dtype='float')

        return tf.where(
            condition=is_inf, x=x, y=(4.0 * (x - min_value) / (max_value - min_value) - 2.0)
        )


class ExponentialNormalization(StatefulLayer):
    """
    Normalization layer based on the exponential moving average of mean and variance over the
    temporal sequence of inputs
    (specification key: `exponential_normalization`).

    Args:
        decay (parameter, 0.0 <= float <= 1.0): Decay rate
            (<span style="color:#C00000"><b>required</b></span>).
        axes (iter[int >= 0]): Normalization axes, excluding batch axis
            (<span style="color:#00C000"><b>default</b></span>: all but last input axes).
        only_mean (bool): Whether to normalize only with respect to mean, not variance
            (<span style="color:#00C000"><b>default</b></span>: false).
        min_variance (float > 0.0): Clip variance lower than minimum
            (<span style="color:#00C000"><b>default</b></span>: 1e-4).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(
        self, *, decay, axes=None, only_mean=False, min_variance=1e-4, name=None, input_spec=None
    ):
        super().__init__(name=name, input_spec=input_spec)

        self.decay = self.submodule(
            name='decay', module=decay, modules=parameter_modules, dtype='float', min_value=0.0,
            max_value=1.0
        )

        if axes is None:
            self.axes = tuple(range(len(self.input_spec.shape) - 1))
        else:
            self.axes = tuple(axes)

        assert not only_mean or min_variance == 1e-4
        self.only_mean = only_mean
        self.min_variance = min_variance

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    def initialize(self):
        super().initialize()

        shape = (1,) + tuple(
            1 if axis in self.axes else dims for axis, dims in enumerate(self.input_spec.shape)
        )

        self.moving_mean = self.variable(
            name='mean', spec=TensorSpec(type='float', shape=shape), initializer='zeros',
            is_trainable=False, is_saved=True
        )

        if not self.only_mean:
            self.moving_variance = self.variable(
                name='variance', spec=TensorSpec(type='float', shape=shape), initializer='ones',
                is_trainable=False, is_saved=True
            )

    @tf_function(num_args=1)
    def apply(self, *, x, independent):
        if independent or self.decay.is_constant(value=1.0):
            mean = self.moving_mean
            if not self.only_mean:
                variance = self.moving_variance

        else:
            zero = tf_util.constant(value=0, dtype='int')
            one_float = tf_util.constant(value=1.0, dtype='float')
            axes = (0,) + tuple(1 + axis for axis in self.axes)

            batch_size = tf_util.cast(x=tf.shape(input=x)[0], dtype='int')
            is_zero_batch = tf.math.equal(x=batch_size, y=zero)

            if self.only_mean:
                def true_fn():
                    return self.moving_mean

                def false_fn():
                    return tf.math.reduce_mean(input_tensor=x, axis=axes, keepdims=True)

                mean = tf.cond(pred=is_zero_batch, true_fn=true_fn, false_fn=false_fn)

            else:
                def true_fn():
                    return self.moving_mean, self.moving_variance

                def false_fn():
                    _mean = tf.math.reduce_mean(input_tensor=x, axis=axes, keepdims=True)
                    deviation = tf.math.squared_difference(x=x, y=_mean)
                    _variance = tf.reduce_mean(input_tensor=deviation, axis=axes, keepdims=True)
                    return _mean, _variance

                mean, variance = tf.cond(pred=is_zero_batch, true_fn=true_fn, false_fn=false_fn)

            if not self.decay.is_constant(value=0.0):
                decay = self.decay.value()
                batch_size = tf_util.cast(x=batch_size, dtype='float')
                # Pow numerically stable since 0.0 <= decay <= 1.0
                decay = tf.math.pow(x=decay, y=batch_size)

                mean = decay * self.moving_mean + (one_float - decay) * mean
                if not self.only_mean:
                    variance = decay * self.moving_variance + (one_float - decay) * variance

            mean = self.moving_mean.assign(value=mean)
            if not self.only_mean:
                variance = self.moving_variance.assign(value=variance)

        if not self.only_mean:
            min_variance = tf_util.constant(value=self.min_variance, dtype='float')
            reciprocal_stddev = tf.math.rsqrt(x=tf.maximum(x=variance, y=min_variance))

        x = x - tf.stop_gradient(input=mean)
        if not self.only_mean:
            x = x * tf.stop_gradient(input=reciprocal_stddev)

        return x


class InstanceNormalization(Layer):
    """
    Instance normalization layer (specification key: `instance_normalization`).

    Args:
        axes (iter[int >= 0]): Normalization axes, excluding batch axis
            (<span style="color:#00C000"><b>default</b></span>: all input axes).
        only_mean (bool): Whether to normalize only with respect to mean, not variance
            (<span style="color:#00C000"><b>default</b></span>: false).
        min_variance (float > 0.0): Clip variance lower than minimum
            (<span style="color:#00C000"><b>default</b></span>: 1e-4).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(
        self, *, axes=None, only_mean=False, min_variance=1e-4, name=None, input_spec=None
    ):
        super().__init__(name=name, input_spec=input_spec)

        if axes is None:
            self.axes = tuple(range(len(self.input_spec.shape)))
        else:
            self.axes = tuple(axes)

        assert not only_mean or min_variance == 1e-4
        self.only_mean = only_mean
        self.min_variance = min_variance

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    @tf_function(num_args=1)
    def apply(self, *, x):
        axes = tuple(1 + axis for axis in self.axes)

        if self.only_mean:
            mean = tf.math.reduce_mean(input_tensor=x, axis=axes, keepdims=True)

            return x - tf.stop_gradient(input=mean)

        else:
            mean, variance = tf.nn.moments(x=x, axes=axes, keepdims=True)

            min_variance = tf_util.constant(value=self.min_variance, dtype='float')
            reciprocal_stddev = tf.math.rsqrt(x=tf.maximum(x=variance, y=min_variance))

            return (x - tf.stop_gradient(input=mean)) * tf.stop_gradient(input=reciprocal_stddev)


class BatchNormalization(Layer):
    """
    Batch normalization layer, generally should only be used for the agent arguments
    `reward_processing[return_processing]` and `reward_processing[advantage_processing]`
    (specification key: `batch_normalization`).

    Args:
        axes (iter[int >= 0]): Normalization axes, excluding batch axis
            (<span style="color:#00C000"><b>default</b></span>: all but last input axes).
        only_mean (bool): Whether to normalize only with respect to mean, not variance
            (<span style="color:#00C000"><b>default</b></span>: false).
        min_variance (float > 0.0): Clip variance lower than minimum
            (<span style="color:#00C000"><b>default</b></span>: 1e-4).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(
        self, *, axes=None, only_mean=False, min_variance=1e-4, name=None, input_spec=None
    ):
        super().__init__(name=name, input_spec=input_spec)

        if axes is None:
            self.axes = tuple(range(len(self.input_spec.shape) - 1))
        else:
            self.axes = tuple(axes)

        assert not only_mean or min_variance == 1e-4
        self.only_mean = only_mean
        self.min_variance = min_variance

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    @tf_function(num_args=1)
    def apply(self, *, x):
        axes = (0,) + tuple(1 + axis for axis in self.axes)

        if self.only_mean:
            mean = tf.math.reduce_mean(input_tensor=x, axis=axes, keepdims=True)

            return x - tf.stop_gradient(input=mean)

        else:
            mean, variance = tf.nn.moments(x=x, axes=axes, keepdims=True)

            min_variance = tf_util.constant(value=self.min_variance, dtype='float')
            reciprocal_stddev = tf.math.rsqrt(x=tf.maximum(x=variance, y=min_variance))

            return (x - tf.stop_gradient(input=mean)) * tf.stop_gradient(input=reciprocal_stddev)
