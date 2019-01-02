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

from tensorforce import TensorforceError
from tensorforce.core.layers import TransformationBase


class Conv1d(TransformationBase):
    """
    1-dimensional convolutional layer.
    """

    def __init__(
        self, name, size, window=3, stride=1, padding='SAME', bias=True, activation='relu',
        dropout=None, use_cudnn_on_gpu=True, input_spec=None, l2_regularization=None,
        summary_labels=None
    ):
        """
        1-dimensional convolution constructor.

        Args:
            window (int > 0): Window size.
            stride (int > 0): Stride size.
            padding ('SAME' | 'VALID'): Padding type.
            use_cudnn_on_gpu: If true, uses cuDNN on GPU.
        """
        self.window = window
        self.stride = stride
        self.padding = padding
        self.use_cudnn_on_gpu = use_cudnn_on_gpu

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            input_spec=input_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )

    def default_input_spec(self):
        return dict(type='float', shape=(0, 0))

    def get_output_spec(self, input_spec):
        if self.stride != 1:
            raise NotImplementedError
        elif self.squeeze:
            input_spec['shape'] = input_spec['shape'][:-1]
        else:
            input_spec['shape'] = input_spec['shape'][:-1] + (self.size,)
        input_spec.pop('min_value', None)
        input_spec.pop('max_value', None)

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        in_size = self.input_spec['shape'][1]
        self.weights = self.add_variable(
            name='weights', dtype='float', shape=(self.window, in_size, self.size),
            is_trainable=True, initializer='random'
        )

    def tf_apply(self, x):
        x = tf.nn.conv1d(
            value=x, filters=self.weights, stride=self.stride, padding=self.padding,
            use_cudnn_on_gpu=self.use_cudnn_on_gpu
        )

        return super().tf_apply(x=x)


class Conv2d(TransformationBase):
    """
    2-dimensional convolutional layer.
    """

    def __init__(
        self, name, size, window=3, stride=1, padding='SAME', dilation=1, bias=True,
        activation='relu', dropout=None, use_cudnn_on_gpu=True, input_spec=None,
        l2_regularization=None, summary_labels=None
    ):
        """
        2-dimensional convolution constructor.

        Args:
            window (int > 0, (int > 0, int > 0)): Window size.
            stride (int > 0, (int > 0, int > 0)): Stride size.
            padding ('SAME' | 'VALID'): Padding type.
            dilation (int > 0, (int > 0, int > 0)): Dilation value.
            use_cudnn_on_gpu: If true, uses cuDNN on GPU.
        """
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
            input_spec=input_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )

    def default_input_spec(self):
        return dict(type='float', shape=(0, 0, 0))

    def get_output_spec(self, input_spec):
        if self.stride != (1, 1, 1, 1):
            raise NotImplementedError
        elif self.squeeze:
            input_spec['shape'] = input_spec['shape'][:-1]
        else:
            input_spec['shape'] = input_spec['shape'][:-1] + (self.size,)
        input_spec.pop('min_value', None)
        input_spec.pop('max_value', None)

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        in_size = self.input_spec['shape'][2]
        self.weights = self.add_variable(
            name='weights', dtype='float', shape=(self.window + (in_size, self.size)),
            is_trainable=True, initializer='random'
        )

    def tf_apply(self, x):
        x = tf.nn.conv2d(
            input=x, filter=self.weights, strides=self.stride, padding=self.padding,
            use_cudnn_on_gpu=self.use_cudnn_on_gpu, dilations=self.dilation
        )

        return super().tf_apply(x=x)
