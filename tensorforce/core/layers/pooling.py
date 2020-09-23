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
from tensorforce.core import TensorSpec, tf_function, tf_util
from tensorforce.core.layers import Layer


class Pooling(Layer):
    """
    Pooling layer (global pooling) (specification key: `pooling`).

    Args:
        reduction ('concat' | 'max' | 'mean' | 'product' | 'sum'): Pooling type
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, reduction, name=None, input_spec=None):
        if reduction not in ('concat', 'max', 'mean', 'product', 'sum'):
            raise TensorforceError.value(name='pooling', argument='reduction', value=reduction)
        self.reduction = reduction

        super().__init__(name=name, input_spec=input_spec)

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    def output_spec(self):
        output_spec = super().output_spec()

        if self.reduction == 'concat':
            output_spec.shape = (output_spec.size,)
        elif self.reduction in ('max', 'mean', 'product', 'sum'):
            output_spec.shape = (output_spec.shape[-1],)

        output_spec.min_value = None
        output_spec.max_value = None

        return output_spec

    @tf_function(num_args=1)
    def apply(self, *, x):
        if self.reduction == 'concat':
            return tf.reshape(tensor=x, shape=(-1, self.output_spec().size))

        elif self.reduction == 'max':
            for _ in range(tf_util.rank(x=x) - 2):
                x = tf.reduce_max(input_tensor=x, axis=1)
            return x

        elif self.reduction == 'mean':
            for _ in range(tf_util.rank(x=x) - 2):
                x = tf.reduce_mean(input_tensor=x, axis=1)
            return x

        elif self.reduction == 'product':
            for _ in range(tf_util.rank(x=x) - 2):
                x = tf.reduce_prod(input_tensor=x, axis=1)
            return x

        elif self.reduction == 'sum':
            for _ in range(tf_util.rank(x=x) - 2):
                x = tf.reduce_sum(input_tensor=x, axis=1)
            return x


class Flatten(Pooling):
    """
    Flatten layer (specification key: `flatten`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, name=None, input_spec=None):
        super().__init__(reduction='concat', name=name, input_spec=input_spec)

    @tf_function(num_args=1)
    def apply(self, *, x):
        if self.input_spec.shape == ():
            return tf.expand_dims(input=x, axis=1)

        else:
            return super().apply(x=x)


class Pool1d(Layer):
    """
    1-dimensional pooling layer (local pooling) (specification key: `pool1d`).

    Args:
        reduction ('average' | 'max'): Pooling type
            (<span style="color:#C00000"><b>required</b></span>).
        window (int > 0): Window size
            (<span style="color:#00C000"><b>default</b></span>: 2).
        stride (int > 0): Stride size
            (<span style="color:#00C000"><b>default</b></span>: 2).
        padding ('same' | 'valid'): Padding type, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__
            (<span style="color:#00C000"><b>default</b></span>: 'same').
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, reduction, window=2, stride=2, padding='same', name=None,input_spec=None):
        self.reduction = reduction

        if isinstance(window, int):
            self.window = (1, 1, window, 1)
        else:
            raise TensorforceError.type(name='Pool1d', argument='window', dtype=type(window))

        if isinstance(stride, int):
            self.stride = (1, 1, stride, 1)
        else:
            raise TensorforceError.type(name='Pool1d', argument='stride', dtype=type(stride))

        self.padding = padding

        super().__init__(name=name, input_spec=input_spec)

    def default_input_spec(self):
        return TensorSpec(type='float', shape=(0, 0))

    def output_spec(self):
        output_spec = super().output_spec()

        if self.padding == 'same':
            output_spec.shape = (np.ceil(output_spec.shape[0] / self.stride[2]), output_spec.shape[1])
        elif self.padding == 'valid':
            output_spec.shape = (
                np.ceil((output_spec.shape[0] - (self.window[2] - 1)) / self.stride[2]),
                output_spec.shape[1]
            )

        return output_spec

    @tf_function(num_args=1)
    def apply(self, *, x):
        x = tf.expand_dims(input=x, axis=1)

        if self.reduction == 'average':
            x = tf.nn.avg_pool(
                input=x, ksize=self.window, strides=self.stride, padding=self.padding.upper()
            )

        elif self.reduction == 'max':
            x = tf.nn.max_pool(
                input=x, ksize=self.window, strides=self.stride, padding=self.padding.upper()
            )

        x = tf.squeeze(input=x, axis=1)

        return x


class Pool2d(Layer):
    """
    2-dimensional pooling layer (local pooling) (specification key: `pool2d`).

    Args:
        reduction ('average' | 'max'): Pooling type
            (<span style="color:#C00000"><b>required</b></span>).
        window (int > 0 | (int > 0, int > 0)): Window size
            (<span style="color:#00C000"><b>default</b></span>: 2).
        stride (int > 0 | (int > 0, int > 0)): Stride size
            (<span style="color:#00C000"><b>default</b></span>: 2).
        padding ('same' | 'valid'): Padding type, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__
            (<span style="color:#00C000"><b>default</b></span>: 'same').
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, reduction, window=2, stride=2, padding='same', name=None,input_spec=None):
        self.reduction = reduction

        if isinstance(window, int):
            self.window = (1, window, window, 1)
        elif len(window) == 2:
            self.window = (1, window[0], window[1], 1)
        else:
            raise TensorforceError.type(name='Pool2d', argument='window', dtype=type(window))

        if isinstance(stride, int):
            self.stride = (1, stride, stride, 1)
        elif len(window) == 2:
            self.stride = (1, stride[0], stride[1], 1)
        else:
            raise TensorforceError.type(name='Pool2d', argument='stride', dtype=type(stride))

        self.padding = padding

        super().__init__(name=name, input_spec=input_spec)

    def default_input_spec(self):
        return TensorSpec(type='float', shape=(0, 0, 0))

    def output_spec(self):
        output_spec = super().output_spec()

        if self.padding == 'same':
            output_spec.shape = (
                np.ceil(output_spec.shape[0] / self.stride[1]),
                np.ceil(output_spec.shape[1] / self.stride[2]),
                output_spec.shape[2]
            )
        elif self.padding == 'valid':
            output_spec.shape = (
                np.ceil((output_spec.shape[0] - (self.window[1] - 1)) / self.stride[1]),
                np.ceil((output_spec.shape[1] - (self.window[2] - 1)) / self.stride[2]),
                output_spec.shape[2]
            )

        return output_spec

    @tf_function(num_args=1)
    def apply(self, *, x):
        if self.reduction == 'average':
            x = tf.nn.avg_pool(
                input=x, ksize=self.window, strides=self.stride, padding=self.padding.upper()
            )

        elif self.reduction == 'max':
            x = tf.nn.max_pool(
                input=x, ksize=self.window, strides=self.stride, padding=self.padding.upper()
            )

        return x
