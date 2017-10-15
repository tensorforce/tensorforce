# Copyright 2017 reinforce.io. All Rights Reserved.
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

"""
Creates various neural network layers. For most layers, these functions use
TF-slim layer types. The purpose of this class is to encapsulate
layer types to mix between layers available in TF-slim and custom implementations.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import Counter
import json
from math import sqrt
import os

import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError, util


def flatten(x, scope='flatten', summary_level=0):
    """Flatten layer.

    Args:
        x: Input tensor

    Returns: Input tensor reshaped to 1d tensor

    """
    with tf.variable_scope(scope):
        x = tf.reshape(tensor=x, shape=(-1, util.prod(x.get_shape().as_list()[1:])))
    return x


def nonlinearity(x, name='relu', scope='nonlinearity', summary_level=0):
    """ Applies a non-linearity to an input and returns the result.

    Args:
        x: Input tensor
        name: String identifier of non-linearity. Options: elu, relu, selu, sigmoid,
        softmax, softplus, tanh

    Returns:

    """
    with tf.variable_scope(scope):
        if name == 'elu':
            x = tf.nn.elu(features=x)
        elif name == 'relu':
            x = tf.nn.relu(features=x)
            if summary_level >= 3:  # summary level 3: layer activations
                non_zero_pct = (tf.cast(tf.count_nonzero(x), tf.float32) / tf.cast(tf.reduce_prod(tf.shape(x)), tf.float32))
                tf.summary.scalar('relu-sparsity', 1.0 - non_zero_pct)
        elif name == 'selu':
            # https://arxiv.org/pdf/1706.02515.pdf
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            negative = alpha * tf.nn.elu(features=x)
            x = scale * tf.where(condition=(x >= 0.0), x=x, y=negative)
        elif name == 'sigmoid':
            x = tf.sigmoid(x=x)
        elif name == 'softmax':
            x = tf.nn.softmax(logits=x)
        elif name == 'softplus':
            x = tf.nn.softplus(features=x)
        elif name == 'tanh':
            x = tf.nn.tanh(x=x)
        else:
            raise TensorForceError('Invalid non-linearity: {}'.format(name))
    return x


def linear(x, size, weights=None, bias=True, l2_regularization=0.0, l1_regularization=0.0, scope='linear',
           summary_level=0):
    """
    Linear layer.

    Args:
        x: Input tensor. Must be rank 2
        size: Neurons in layer
        weights: None for random matrix, otherwise given float or array is used.
        bias: Bool to indicate whether bias is used, otherwise given float or array is used.
        l2_regularization: L2-regularisation value
        weights: Weights for layer. If none, initialisation defaults to Xavier (normal with
        size/shape dependent standard deviation).

    Returns:

    """
    input_rank = util.rank(x)
    if input_rank != 2:
        raise TensorForceError('Invalid input rank for linear layer: {},'
                               ' must be 2.'.format(input_rank))

    with tf.variable_scope(scope):
        weights_shape = (x.shape[1].value, size)

        if weights is None:
            stddev = min(0.1, sqrt(2.0 / (x.shape[1].value + size)))
            weights_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)

        elif isinstance(weights, float):
            if weights == 0.0:
                weights_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                weights_init = tf.constant_initializer(value=weights, dtype=tf.float32)

        elif isinstance(weights, tf.Tensor):
            if util.shape(weights) != weights_shape:
                raise TensorForceError(
                    'Weights shape {} does not match expected shape {} '.format(weights.shape, weights_shape)
                )
            weights_init = weights

        else:
            weights = np.asarray(weights, dtype=np.float32)
            if weights.shape != weights_shape:
                raise TensorForceError(
                    'Weights shape {} does not match expected shape {} '.format(weights.shape, weights_shape)
                )
            weights_init = tf.constant_initializer(value=weights, dtype=tf.float32)

        bias_shape = (size,)

        if isinstance(bias, bool):
            if bias:
                bias_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                bias_init = None

        elif isinstance(bias, float):
            if bias == 0.0:
                bias_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                bias_init = tf.constant_initializer(value=bias, dtype=tf.float32)

        elif isinstance(bias, tf.Tensor):
            if util.shape(bias) != bias_shape:
                raise TensorForceError(
                    'Bias shape {} does not match expected shape {} '.format(bias.shape, bias_shape)
                )
            bias_init = bias

        else:
            bias = np.asarray(bias, dtype=np.float32)
            if bias.shape != bias_shape:
                raise TensorForceError(
                    'Bias shape {} does not match expected shape {} '.format(bias.shape, bias_shape)
                )
            bias_init = tf.constant_initializer(value=bias, dtype=tf.float32)

        if isinstance(weights_init, tf.Tensor):
            weights = weights_init
        else:
            weights = tf.get_variable(name='W', shape=weights_shape, dtype=tf.float32, initializer=weights_init)

        if l2_regularization > 0.0:
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=weights))
        if l1_regularization > 0.0:
            weight_l1_t = tf.convert_to_tensor(l1_regularization, dtype=tf.float32, name='weight_l1')
            tf.losses.add_loss(tf.multiply(weight_l1_t, tf.reduce_sum(tf.abs(weights)), name='loss_l1_weights'))
        else:
            weight_l1_t = None

        x = tf.matmul(a=x, b=weights)

        if bias_init is not None:
            if isinstance(bias_init, tf.Tensor):
                bias = bias_init
            else:
                bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)

            if l2_regularization > 0.0:
                tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=bias))
            if l1_regularization > 0.0:
                tf.losses.add_loss(tf.multiply(weight_l1_t, tf.reduce_sum(tf.abs(bias)), name='loss_l1_bias'))

            x = tf.nn.bias_add(value=x, bias=bias)

    return x


def dense(x, size, bias=True, activation='relu', l2_regularization=0.0, l1_regularization=0.0, scope='dense',
          summary_level=0):
    """
    Fully connected layer.

    Args:
        x: Input tensor
        size: Neurons in layer
        bias: Bool, indicates whether bias is used
        activation: Non-linearity type, defaults to relu
        l2_regularization: L2-regularisation value

    Returns:

    """
    input_rank = util.rank(x)
    if input_rank != 2:
        raise TensorForceError('Invalid input rank for linear layer: {},'
                               ' must be 2.'.format(input_rank))

    with tf.variable_scope(scope):
        x = linear(x=x, size=size, bias=bias, l2_regularization=l2_regularization, l1_regularization=l1_regularization)
        x = nonlinearity(x=x, name=activation, summary_level=summary_level)

        if summary_level >= 3:
            tf.summary.histogram('activations', x)
    return x


def conv1d(x, size, window=3, stride=1, padding='SAME', bias=False, activation='relu',
           l2_regularization=0.0, scope='conv1d', summary_level=0):
    """A 1d convolutional layer.
    Args:
        x: Input tensor. Must be rank 3
        size: Neurons
        window: Filter window size
        stride: Filter stride
        padding: One of [VALID, SAME]
        bias: Bool, indicates whether bias is used
        activation: Non-linearity type, defaults to relu
        l2_regularization: L2-regularisation value
    Returns:
    """
    input_rank = util.rank(x)
    if input_rank != 3:
        raise TensorForceError('Invalid input rank for conv1d layer: {}, must be 3'.format(input_rank))

    with tf.variable_scope(scope):
        filters_shape = (window, x.shape[2].value, size)
        stddev = min(0.1, sqrt(2.0 / size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)

        if l2_regularization > 0.0:
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=filters))

        x = tf.nn.conv1d(value=x, filters=filters, stride=stride, padding=padding)

        if bias:
            bias_shape = (size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)

            if l2_regularization > 0.0:
                tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=bias))

            x = tf.nn.bias_add(value=x, bias=bias)

        x = nonlinearity(x=x, name=activation, summary_level=summary_level)

        if summary_level >= 3:
            tf.summary.histogram('activations', x)
    return x


def conv2d(x, size, window=3, stride=1, padding='SAME', bias=False, activation='relu',
           l2_regularization=0.0, scope='conv2d', summary_level=0):
    """A 2d convolutional layer.

    Args:
        x: Input tensor. Must be rank 4
        size: Neurons
        window: Filter window size
        stride: Filter stride
        padding: One of [VALID, SAME]
        bias: Bool, indicates whether bias is used
        activation: Non-linearity type, defaults to relu
        l2_regularization: L2-regularisation value

    Returns:

    """
    input_rank = util.rank(x)
    if input_rank != 4:
        raise TensorForceError('Invalid input rank for conv2d layer: {}, must be 4'.format(input_rank))

    with tf.variable_scope(scope):
        filters_shape = (window, window, x.shape[3].value, size)
        stddev = min(0.1, sqrt(2.0 / size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)

        if l2_regularization > 0.0:
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=filters))

        x = tf.nn.conv2d(input=x, filter=filters, strides=(1, stride, stride, 1), padding=padding)

        if bias:
            bias_shape = (size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)

            if l2_regularization > 0.0:
                tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=bias))

            x = tf.nn.bias_add(value=x, bias=bias)

        x = nonlinearity(x=x, name=activation, summary_level=summary_level)

        if summary_level >= 3:
            tf.summary.histogram('activations', x)
    return x


def lstm(x, size=None, dropout=None, scope='lstm', summary_level=0):
    """

    Args:
        x: Input tensor.
        size: Layer size, defaults to input size.
        dropout: dropout_keep_prob (eg 0.5) for regularization, applied via rnn.DropoutWrapper

    Returns:

    """
    input_rank = util.rank(x)
    if input_rank != 2:
        raise TensorForceError('Invalid input rank for lstm layer: {},'
                               ' must be 2.'.format(input_rank))
    if not size:
        size = x.get_shape()[1].value

    with tf.variable_scope(scope):
        internal_input = tf.placeholder(dtype=tf.float32, shape=(None, 2, size))
        lstm_cell = tf.contrib.rnn.LSTMCell(num_units=size)
        if dropout:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1 - dropout)
        c = internal_input[:, 0, :]
        h = internal_input[:, 1, :]
        state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
        x, state = lstm_cell(inputs=x, state=state)

        internal_output = tf.stack(values=(state.c, state.h), axis=1)
        internal_init = np.zeros(shape=(2, size))

        if summary_level >= 3:
            tf.summary.histogram('activations', x)
    return x, (internal_input,), (internal_output,), (internal_init,)


layers = {
    'flatten': flatten,
    'nonlinearity': nonlinearity,
    'linear': linear,
    'dense': dense,
    'conv1d': conv1d,
    'conv2d': conv2d,
    'lstm': lstm
}


def layered_network_builder(layers_config):
    """Returns a function defining a layered neural network according to the given configuration.


    Args:
        layers_config: Iterable of layer configuration dicts.

    Returns:

    """

    def network_builder(inputs, summary_level=0):
        input_length = len(inputs)

        if input_length != 1:
            raise TensorForceError('Layered network must have only one input,'
                                   ' input length {} given.'.format(input_length))
        x = next(iter(inputs.values()))
        internal_inputs = []
        internal_outputs = []
        internal_inits = []

        layer_counter = Counter()
        for layer_config in layers_config:
            if callable(layer_config['type']):
                scope = layer_config['type'].__name__ + str(layer_counter[layer_config['type']])
            else:
                scope = layer_config['type'] + str(layer_counter[layer_config['type']])

            x = util.get_object(
                obj=layer_config,
                predefined=layers,
                kwargs=dict(x=x, scope=scope, summary_level=summary_level)
            )
            layer_counter[layer_config['type']] += 1
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 4
                internal_inputs.extend(x[1])
                internal_outputs.extend(x[2])
                internal_inits.extend(x[3])
                x = x[0]

        if internal_inputs:
            return x, internal_inputs, internal_outputs, internal_inits
        else:
            return x

    return network_builder


def from_json(filename):
    """Creates a layer_networkd_builder from a JSON.

    Args:
        filename: Path to configuration

    Returns: A layered_network_builder function with layers generated from the JSON

    """
    path = os.path.join(os.getcwd(), filename)
    with open(path, 'r') as fp:
        config = json.load(fp=fp)

    return layered_network_builder(config)
