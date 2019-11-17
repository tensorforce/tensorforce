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
from tensorforce.core import Module, parameter_modules
from tensorforce.core.layers import Layer


class ExponentialNormalization(Layer):
    """
    Normalization layer based on the exponential moving average (specification key:
    `exponential_normalization`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        decay (parameter, 0.0 <= float <= 1.0): Decay rate
            (<span style="color:#00C000"><b>default</b></span>: 0.999).
        axes (iter[int >= 0]): Normalization axes, excluding batch axis
            (<span style="color:#00C000"><b>default</b></span>: all but last axis).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, decay=0.999, axes=None, input_spec=None, summary_labels=None):
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

        self.decay = self.add_module(
            name='decay', module=decay, modules=parameter_modules, dtype='float'
        )

        self.axes = axes if axes is None else tuple(axes)

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def tf_initialize(self):
        super().tf_initialize()

        shape = self.input_spec['shape']
        if self.axes is None:
            if len(shape) > 0:
                self.axes = tuple(range(len(shape) - 1))
                shape = tuple(1 for _ in shape[:-1]) + (shape[-1],)
            else:
                self.axes = ()
        else:
            shape = tuple(1 if axis in self.axes else dims for axis, dims in enumerate(shape))
        shape = (1,) + shape

        self.moving_mean = self.add_variable(
            name='mean', dtype='float', shape=shape, is_trainable=False, initializer='zeros'
        )

        self.moving_variance = self.add_variable(
            name='variance', dtype='float', shape=shape, is_trainable=False, initializer='zeros'
        )

        self.after_first_call = self.add_variable(
            name='after-first-call', dtype='bool', shape=(), is_trainable=False,
            initializer='zeros'
        )

        self.update_on_optimization = self.add_variable(
            name='update-on-optimization', dtype='bool', shape=(), is_trainable=False,
            initializer='zeros'
        )

    def tf_apply(self, x):

        def no_update():
            return self.moving_mean, self.moving_variance

        def apply_update():
            one = tf.constant(value=1.0, dtype=util.tf_dtype(dtype='float'))
            axes = tuple(1 + axis for axis in self.axes)

            decay = self.decay.value()
            batch_size = tf.dtypes.cast(x=tf.shape(input=x)[0], dtype=util.tf_dtype(dtype='float'))
            decay = tf.math.pow(x=decay, y=batch_size)

            mean = tf.math.reduce_mean(input_tensor=x, axis=axes, keepdims=True)
            mean = tf.where(
                condition=self.after_first_call,
                x=(decay * self.moving_mean + (one - decay) * mean), y=mean
            )

            variance = tf.reduce_mean(
                input_tensor=tf.math.squared_difference(x=x, y=mean), axis=axes, keepdims=True
            )
            variance = tf.where(
                condition=self.after_first_call,
                x=(decay * self.moving_variance + (one - decay) * variance), y=variance
            )

            with tf.control_dependencies(control_inputs=(mean, variance)):
                assignment = self.after_first_call.assign(
                    value=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')),
                    read_value=False
                )

            with tf.control_dependencies(control_inputs=(assignment,)):
                variance = self.moving_variance.assign(value=variance)
                mean = self.moving_mean.assign(value=mean)

            return mean, variance

        optimization = Module.retrieve_tensor(name='optimization')
        update_on_optimization = tf.where(
            condition=self.after_first_call, x=self.update_on_optimization, y=optimization
        )
        update_on_optimization = self.update_on_optimization.assign(value=update_on_optimization)
        skip_update = tf.math.logical_or(
            x=Module.retrieve_tensor(name='independent'),
            y=tf.math.not_equal(x=update_on_optimization, y=optimization)
        )

        mean, variance = self.cond(pred=skip_update, true_fn=no_update, false_fn=apply_update)

        epsilon = tf.constant(value=util.epsilon, dtype=util.tf_dtype(dtype='float'))
        reciprocal_stddev = tf.math.rsqrt(x=tf.maximum(x=variance, y=epsilon))

        x = (x - tf.stop_gradient(input=mean)) * tf.stop_gradient(input=reciprocal_stddev)

        return x


class InstanceNormalization(Layer):
    """
    Instance normalization layer (specification key: `instance_normalization`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        axes (iter[int >= 0]): Normalization axes, excluding batch axis
            (<span style="color:#00C000"><b>default</b></span>: all).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, axes=None, input_spec=None, summary_labels=None):
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

        self.axes = axes if axes is None else tuple(axes)

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def tf_apply(self, x):
        epsilon = tf.constant(value=util.epsilon, dtype=util.tf_dtype(dtype='float'))

        if self.axes is None:
            mean, variance = tf.nn.moments(
                x=x, axes=tuple(range(1, len(self.input_spec['shape']))), keepdims=True
            )
        else:
            mean, variance = tf.nn.moments(
                x=x, axes=tuple(1 + axis for axis in self.axes), keepdims=True
            )

        reciprocal_stddev = tf.math.rsqrt(x=tf.maximum(x=variance, y=epsilon))

        x = (x - tf.stop_gradient(input=mean)) * tf.stop_gradient(input=reciprocal_stddev)

        return x
