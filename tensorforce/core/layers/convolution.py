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
from tensorflow.python.keras.utils.conv_utils import conv_output_length, deconv_output_length

from tensorforce import TensorforceError, util
from tensorforce.core import TensorSpec, tf_function, tf_util
from tensorforce.core.layers import TransformationBase


class Conv1d(TransformationBase):
    """
    1-dimensional convolutional layer (specification key: `conv1d`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        window (int > 0): Window size
            (<span style="color:#00C000"><b>default</b></span>: 3).
        stride (int > 0): Stride size
            (<span style="color:#00C000"><b>default</b></span>: 1).
        padding ('same' | 'valid'): Padding type, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__
            (<span style="color:#00C000"><b>default</b></span>: 'same').
        dilation (int > 0 | (int > 0, int > 0)): Dilation value
            (<span style="color:#00C000"><b>default</b></span>: 1).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: relu).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        initialization_scale (float > 0.0): Initialization scale
            (<span style="color:#00C000"><b>default</b></span>: 1.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(
        self, *, size, window=3, stride=1, padding='same', dilation=1, bias=True, activation='relu',
        dropout=0.0, initialization_scale=1.0, vars_trainable=True, l2_regularization=None,
        name=None, input_spec=None
    ):
        self.window = window
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        super().__init__(
            size=size, bias=bias, activation=activation, dropout=dropout,
            vars_trainable=vars_trainable,  l2_regularization=l2_regularization, name=name,
            input_spec=input_spec
        )

        self.initialization_scale = initialization_scale

    def default_input_spec(self):
        return TensorSpec(type='float', shape=(0, 0))

    def output_spec(self):
        output_spec = super().output_spec()

        length = conv_output_length(
            input_length=output_spec.shape[0], filter_size=self.window, padding=self.padding,
            stride=self.stride, dilation=self.dilation
        )

        if self.squeeze:
            output_spec.shape = (length,)
        else:
            output_spec.shape = (length, self.size)

        output_spec.min_value = None
        output_spec.max_value = None

        return output_spec

    def initialize(self):
        super().initialize()

        in_size = self.input_spec.shape[1]

        initializer = 'orthogonal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        self.weights = self.variable(
            name='weights', spec=TensorSpec(type='float', shape=(self.window, in_size, self.size)),
            initializer=initializer, initialization_scale=self.initialization_scale,
            is_trainable=self.vars_trainable, is_saved=True
        )

    @tf_function(num_args=1)
    def apply(self, *, x):
        x = tf.nn.conv1d(
            input=x, filters=self.weights, stride=self.stride, padding=self.padding.upper(),
            dilations=self.dilation
        )

        return super().apply(x=x)


class Conv2d(TransformationBase):
    """
    2-dimensional convolutional layer (specification key: `conv2d`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        window (int > 0 | (int > 0, int > 0)): Window size
            (<span style="color:#00C000"><b>default</b></span>: 3).
        stride (int > 0 | (int > 0, int > 0)): Stride size
            (<span style="color:#00C000"><b>default</b></span>: 1).
        padding ('same' | 'valid'): Padding type, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__
            (<span style="color:#00C000"><b>default</b></span>: 'same').
        dilation (int > 0 | (int > 0, int > 0)): Dilation value
            (<span style="color:#00C000"><b>default</b></span>: 1).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: "relu").
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        initialization_scale (float > 0.0): Initialization scale
            (<span style="color:#00C000"><b>default</b></span>: 1.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(
        self, *, size, window=3, stride=1, padding='same', dilation=1, bias=True, activation='relu',
        dropout=0.0, initialization_scale=1.0, vars_trainable=True, l2_regularization=None,
        name=None, input_spec=None
    ):
        if isinstance(window, int):
            self.window = (window, window)
        elif util.is_iterable(x=window) and len(window) == 2:
            self.window = tuple(window)
        else:
            raise TensorforceError.type(name='Conv2d', argument='window', dtype=type(window))

        if isinstance(stride, int):
            self.stride = (1, stride, stride, 1)
        elif util.is_iterable(x=stride) and len(stride) == 2:
            self.stride = (1, stride[0], stride[1], 1)
        else:
            raise TensorforceError.type(name='Conv2d', argument='stride', dtype=type(stride))

        self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (1, dilation, dilation, 1)
        elif util.is_iterable(x=dilation) and len(dilation) == 2:
            self.dilation = (1, dilation[0], dilation[1], 1)
        else:
            raise TensorforceError.type(name='Conv2d', argument='dilation', dtype=type(dilation))

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            vars_trainable=vars_trainable, input_spec=input_spec,
            l2_regularization=l2_regularization
        )

        self.initialization_scale = initialization_scale

    def default_input_spec(self):
        return TensorSpec(type='float', shape=(0, 0, 0))

    def output_spec(self):
        output_spec = super().output_spec()

        height = conv_output_length(
            input_length=output_spec.shape[0], filter_size=self.window[0], padding=self.padding,
            stride=self.stride[1], dilation=self.dilation[1]
        )
        width = conv_output_length(
            input_length=output_spec.shape[1], filter_size=self.window[1], padding=self.padding,
            stride=self.stride[2], dilation=self.dilation[2]
        )

        if self.squeeze:
            output_spec.shape = (height, width)
        else:
            output_spec.shape = (height, width, self.size)

        output_spec.min_value = None
        output_spec.max_value = None

        return output_spec

    def initialize(self):
        super().initialize()

        in_size = self.input_spec.shape[2]

        initializer = 'orthogonal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        self.weights = self.variable(
            name='weights',
            spec=TensorSpec(type='float', shape=(self.window + (in_size, self.size))),
            initializer=initializer, initialization_scale=self.initialization_scale,
            is_trainable=self.vars_trainable, is_saved=True
        )

    @tf_function(num_args=1)
    def apply(self, *, x):
        x = tf.nn.conv2d(
            input=x, filters=self.weights, strides=self.stride, padding=self.padding.upper(),
            dilations=self.dilation
        )

        return super().apply(x=x)


class Conv1dTranspose(TransformationBase):
    """
    1-dimensional transposed convolutional layer, also known as deconvolution layer
    (specification key: `deconv1d`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        window (int > 0): Window size
            (<span style="color:#00C000"><b>default</b></span>: 3).
        output_width (int > 0): Output width
            (<span style="color:#00C000"><b>default</b></span>: same as input).
        stride (int > 0): Stride size
            (<span style="color:#00C000"><b>default</b></span>: 1).
        padding ('same' | 'valid'): Padding type, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__
            (<span style="color:#00C000"><b>default</b></span>: 'same').
        dilation (int > 0 | (int > 0, int > 0)): Dilation value
            (<span style="color:#00C000"><b>default</b></span>: 1).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: "relu").
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        initialization_scale (float > 0.0): Initialization scale
            (<span style="color:#00C000"><b>default</b></span>: 1.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(
        self, *, size, window=3, output_width=None, stride=1, padding='same', dilation=1, bias=True,
        activation='relu', dropout=0.0, initialization_scale=1.0, vars_trainable=True,
        l2_regularization=None, name=None, input_spec=None
    ):
        self.window = window
        if output_width is None:
            self.output_shape = None
        elif self.squeeze:
            self.output_shape = (output_width, max(1, size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            vars_trainable=vars_trainable, input_spec=input_spec,
            l2_regularization=l2_regularization
        )

        self.initialization_scale = initialization_scale

    def default_input_spec(self):
        return TensorSpec(type='float', shape=(0, 0))

    def output_spec(self):
        output_spec = super().output_spec()

        width = deconv_output_length(
            input_length=output_spec.shape[0], filter_size=self.window, padding=self.padding,
            stride=self.stride, dilation=self.dilation
        )

        if self.output_shape is None:
            self.output_shape = (width, self.size)

        if self.squeeze:
            output_spec.shape = self.output_shape[:1]
        else:
            output_spec.shape = self.output_shape

        output_spec.min_value = None
        output_spec.max_value = None

        return output_spec

    def initialize(self):
        super().initialize()

        in_size = self.input_spec.shape[1]

        initializer = 'orthogonal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        self.weights = self.variable(
            name='weights', spec=TensorSpec(type='float', shape=(self.window, in_size, self.size)),
            initializer=initializer, initialization_scale=self.initialization_scale,
            is_trainable=self.vars_trainable, is_saved=True
        )

    @tf_function(num_args=1)
    def apply(self, *, x):
        output_shape = tf.concat(values=[
            tf_util.cast(x=tf.shape(input=x)[:1], dtype='int'),
            tf_util.constant(value=self.output_shape, dtype='int')
        ], axis=0)
        x = tf.nn.conv1d_transpose(
            input=x, filters=self.weights, output_shape=tf_util.int32(x=output_shape),
            strides=self.stride, padding=self.padding.upper(), dilations=self.dilation
        )

        return super().apply(x=x)


class Conv2dTranspose(TransformationBase):
    """
    2-dimensional transposed convolutional layer, also known as deconvolution layer
    (specification key: `deconv2d`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        window (int > 0 | (int > 0, int > 0)): Window size
            (<span style="color:#00C000"><b>default</b></span>: 3).
        output_shape (int > 0 | (int > 0, int > 0)): Output shape
            (<span style="color:#00C000"><b>default</b></span>: same as input).
        stride (int > 0 | (int > 0, int > 0)): Stride size
            (<span style="color:#00C000"><b>default</b></span>: 1).
        padding ('same' | 'valid'): Padding type, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__
            (<span style="color:#00C000"><b>default</b></span>: 'same').
        dilation (int > 0 | (int > 0, int > 0)): Dilation value
            (<span style="color:#00C000"><b>default</b></span>: 1).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: "relu").
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        initialization_scale (float > 0.0): Initialization scale
            (<span style="color:#00C000"><b>default</b></span>: 1.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(
        self, *, size, window=3, output_shape=None, stride=1, padding='same', dilation=1, bias=True,
        activation='relu', dropout=0.0, initialization_scale=1.0, vars_trainable=True,
        l2_regularization=None, name=None, input_spec=None
    ):
        if isinstance(window, int):
            self.window = (window, window)
        elif util.is_iterable(x=window) and len(window) == 2:
            self.window = tuple(window)
        else:
            raise TensorforceError.type(
                name='Conv2dTranspose', argument='window', dtype=type(window)
            )

        if output_shape is None:
            self.output_shape = None
        elif isinstance(output_shape, int):
            self.output_shape = (output_shape, output_shape, max(1, size))
        elif util.is_iterable(x=window) and len(output_shape) == 2:
            self.output_shape = (output_shape[0], output_shape[1], max(1, size))
        else:
            raise TensorforceError.type(
                name='Conv2dTranspose', argument='window', dtype=type(output_shape)
            )

        if isinstance(stride, int):
            self.stride = (1, stride, stride, 1)
        elif util.is_iterable(x=stride) and len(stride) == 2:
            self.stride = (1, stride[0], stride[1], 1)
        else:
            raise TensorforceError.type(
                name='Conv2dTranspose', argument='stride', dtype=type(stride)
            )

        self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (1, dilation, dilation, 1)
        elif len(dilation) == 2:
            self.dilation = (1, dilation[0], dilation[1], 1)
        else:
            raise TensorforceError.type(
                name='Conv2dTranspose', argument='dilation', dtype=type(dilation)
            )

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            vars_trainable=vars_trainable, input_spec=input_spec,
            l2_regularization=l2_regularization
        )

        self.initialization_scale = initialization_scale

    def default_input_spec(self):
        return TensorSpec(type='float', shape=(0, 0, 0))

    def output_spec(self):
        output_spec = super().output_spec()

        height = deconv_output_length(
            input_length=output_spec.shape[0], filter_size=self.window[0], padding=self.padding,
            stride=self.stride[1], dilation=self.dilation[1]
        )
        width = deconv_output_length(
            input_length=output_spec.shape[1], filter_size=self.window[1], padding=self.padding,
            stride=self.stride[2], dilation=self.dilation[2]
        )

        if self.output_shape is None:
            self.output_shape = (height, width, self.size)

        if self.squeeze:
            output_spec.shape = self.output_shape[: 2]
        else:
            output_spec.shape = self.output_shape

        output_spec.min_value = None
        output_spec.max_value = None

        return output_spec

    def initialize(self):
        super().initialize()

        in_size = self.input_spec.shape[2]

        initializer = 'orthogonal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        self.weights = self.variable(
            name='weights',
            spec=TensorSpec(type='float', shape=(self.window + (in_size, self.size))),
            initializer=initializer, initialization_scale=self.initialization_scale,
            is_trainable=self.vars_trainable, is_saved=True
        )

    @tf_function(num_args=1)
    def apply(self, *, x):
        output_shape = tf.concat(values=[
            tf_util.cast(x=tf.shape(input=x)[:1], dtype='int'),
            tf_util.constant(value=self.output_shape, dtype='int')
        ], axis=0)
        x = tf.nn.conv2d_transpose(
            input=x, filters=self.weights, output_shape=tf_util.int32(x=output_shape),
            strides=self.stride, padding=self.padding.upper(), dilations=self.dilation
        )

        return super().apply(x=x)
