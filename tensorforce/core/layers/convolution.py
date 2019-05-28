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

from math import ceil

import tensorflow as tf

from tensorforce import TensorforceError
from tensorforce.core.layers import TransformationBase


class Conv1d(TransformationBase):
    """
    1-dimensional convolutional layer (specification key: `conv1d`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        window (int > 0): Window size
            (<span style="color:#00C000"><b>default</b></span>: 3).
        stride (int > 0): Stride size
            (<span style="color:#00C000"><b>default</b></span>: 1).
        padding ('same' | 'valid'): Padding type, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__
            (<span style="color:#00C000"><b>default</b></span>: 'same').
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: "relu").
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        is_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        use_cudnn_on_gpu (bool): Whether to use cuDNN on GPU
            (<span style="color:#00C000"><b>default</b></span>: true).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, size, window=3, stride=1, padding='same', bias=True, activation='relu',
        dropout=0.0, is_trainable=True, use_cudnn_on_gpu=True, input_spec=None,
        summary_labels=None, l2_regularization=None
    ):
        self.window = window
        self.stride = stride
        self.padding = padding

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            is_trainable=is_trainable, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.use_cudnn_on_gpu = use_cudnn_on_gpu

    def default_input_spec(self):
        return dict(type='float', shape=(0, 0))

    def get_output_spec(self, input_spec):
        if self.padding == 'same':
            shape = (ceil(input_spec['shape'][0] / self.stride),)
        elif self.padding == 'valid':
            shape = (ceil((input_spec['shape'][0] - (self.window - 1)) / self.stride),)

        if self.squeeze:
            input_spec['shape'] = shape
        else:
            input_spec['shape'] = shape + (self.size,)

        input_spec.pop('min_value', None)
        input_spec.pop('max_value', None)

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        in_size = self.input_spec['shape'][1]

        initializer = 'orthogonal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        self.weights = self.add_variable(
            name='weights', dtype='float', shape=(self.window, in_size, self.size),
            is_trainable=self.is_trainable, initializer=initializer
        )

    def tf_apply(self, x):
        x = tf.nn.conv1d(
            value=x, filters=self.weights, stride=self.stride, padding=self.padding.upper(),
            use_cudnn_on_gpu=self.use_cudnn_on_gpu
        )

        return super().tf_apply(x=x)


class Conv2d(TransformationBase):
    """
    2-dimensional convolutional layer (specification key: `conv2d`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
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
        is_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        use_cudnn_on_gpu (bool): Whether to use cuDNN on GPU
            (<span style="color:#00C000"><b>default</b></span>: true).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, size, window=3, stride=1, padding='same', dilation=1, bias=True,
        activation='relu', dropout=0.0, is_trainable=True, use_cudnn_on_gpu=True, input_spec=None,
        summary_labels=None, l2_regularization=None
    ):
        if isinstance(window, int):
            self.window = (window, window)
        elif len(window) == 2:
            self.window = tuple(window)
        else:
            raise TensorforceError("Invalid window argument for conv2d layer: {}.".format(window))
        if isinstance(stride, int):
            self.stride = (1, stride, stride, 1)
        elif len(stride) == 2:
            self.stride = (1, stride[0], stride[1], 1)
        else:
            raise TensorforceError("Invalid stride argument for conv2d layer: {}.".format(stride))
        self.padding = padding
        if isinstance(dilation, int):
            self.dilation = (1, dilation, dilation, 1)
        elif len(dilation) == 2:
            self.dilation = (1, dilation[0], dilation[1], 1)
        else:
            raise TensorforceError("Invalid dilation argument for conv2d layer: {}.".format(dilation))
        self.use_cudnn_on_gpu = use_cudnn_on_gpu

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            is_trainable=is_trainable, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

    def default_input_spec(self):
        return dict(type='float', shape=(0, 0, 0))

    def get_output_spec(self, input_spec):
        if self.padding == 'same':
            shape = (
                ceil(input_spec['shape'][0] / self.stride[1]),
                ceil(input_spec['shape'][1] / self.stride[2])
            )
        elif self.padding == 'valid':
            shape = (
                ceil((input_spec['shape'][0] - (self.window[1] - 1)) / self.stride[1]),
                ceil((input_spec['shape'][1] - (self.window[2] - 1)) / self.stride[2])
            )

        if self.squeeze:
            input_spec['shape'] = shape
        else:
            input_spec['shape'] = shape + (self.size,)

        input_spec.pop('min_value', None)
        input_spec.pop('max_value', None)

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        in_size = self.input_spec['shape'][2]

        initializer = 'orthogonal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        self.weights = self.add_variable(
            name='weights', dtype='float', shape=(self.window + (in_size, self.size)),
            is_trainable=self.is_trainable, initializer=initializer
        )

    def tf_apply(self, x):
        x = tf.nn.conv2d(
            input=x, filter=self.weights, strides=self.stride, padding=self.padding.upper(),
            use_cudnn_on_gpu=self.use_cudnn_on_gpu, dilations=self.dilation
        )

        return super().tf_apply(x=x)
