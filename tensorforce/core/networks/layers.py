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

import json
from math import sqrt
import os

import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError, util


def flatten(x):
    with tf.variable_scope('flatten'):
        x = tf.reshape(tensor=x, shape=(-1, util.prod(x.get_shape().as_list()[1:])))
    return x


def nonlinearity(x, name='relu'):
    with tf.variable_scope('nonlinearity'):
        if name == 'elu':
            x = tf.nn.elu(features=x)
        elif name == 'relu':
            x = tf.nn.relu(features=x)
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
        elif name == 'tanh':
            x = tf.nn.tanh(x=x)
        else:
            raise TensorForceError('Invalid nonlinearity.')
    return x


def linear(x, size, bias=True, l2_regularization=0.0):
    if util.rank(x) != 2:
        raise TensorForceError('Invalid input rank for linear layer.')
    with tf.variable_scope('linear'):
        weights = tf.Variable(initial_value=tf.random_normal(shape=(x.get_shape()[1].value, size), stddev=sqrt(2.0 / (x.get_shape()[1].value + size))))
        if l2_regularization > 0.0:
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=weights))
        x = tf.matmul(a=x, b=weights)
        if bias:
            bias = tf.Variable(initial_value=tf.zeros(shape=(size,)))
            if l2_regularization > 0.0:
                tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=bias))
            x = tf.nn.bias_add(value=x, bias=bias)
    return x


def dense(x, size, bias=True, activation='relu', l2_regularization=0.0):
    if util.rank(x) != 2:
        raise TensorForceError('Invalid input rank for dense layer.')
    with tf.variable_scope('dense'):
        x = linear(x=x, size=size, bias=bias, l2_regularization=l2_regularization)
        x = nonlinearity(x=x, name=activation)
    return x


def conv2d(x, size, window=3, stride=1, bias=False, activation='relu', l2_regularization=0.0):
    if util.rank(x) != 4:
        raise TensorForceError('Invalid input rank for conv2d layer.')
    with tf.variable_scope('conv2d'):
        filters = tf.Variable(initial_value=tf.random_normal(shape=(window, window, x.get_shape()[3].value, size), stddev=sqrt(2.0 / size)))
        if l2_regularization > 0.0:
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=filters))
        x = tf.nn.conv2d(input=x, filter=filters, strides=(1, stride, stride, 1), padding='SAME')
        if bias:
            bias = tf.Variable(initial_value=tf.zeros(shape=(size,)))
            if l2_regularization > 0.0:
                tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=bias))
            x = tf.nn.bias_add(value=x, bias=bias)
        x = nonlinearity(x=x, name=activation)
    return x


def lstm(x, size=None):
    """
    Creates an LSTM layer.
    """
    if util.rank(x) != 2:
        raise TensorForceError('Invalid input rank for lstm layer.')
    if not size:
        size = x.get_shape()[1].value
    with tf.variable_scope('lstm'):
        internal_input = tf.placeholder(dtype=tf.float32, shape=(None, 2, size))
        lstm = tf.contrib.rnn.LSTMCell(num_units=size)
        state = tf.contrib.rnn.LSTMStateTuple(c=internal_input[:, 0, :], h=internal_input[:, 1, :])
        x, state = lstm(inputs=x, state=state)
        internal_output = tf.stack(values=(state.c, state.h), axis=1)
        internal_init = np.zeros(shape=(2, size))
    return x, (internal_input,), (internal_output,), (internal_init,)


layers = {
    'flatten': flatten,
    'nonlinearity': nonlinearity,
    'linear': linear,
    'dense': dense,
    'conv2d': conv2d,
    'lstm': lstm
}


def layered_network_builder(layers_config):
    """
    Returns a function defining a layered neural network according to the given configuration.

    :param layers: Dict that describes a neural network layer-wise
    :return: A function defining a TensorFlow network
    """

    def network_builder(inputs):
        if len(inputs) != 1:
            raise TensorForceError('Layered network must have only one input.')
        x = next(iter(inputs.values()))
        internal_inputs = []
        internal_outputs = []
        internal_inits = []

        for layer_config in layers_config:
            layer_type = layer_config['type']
            layer = util.function(layer_type, predefined=layers)
            x = layer(x=x, **{k: v for k, v in layer_config.items() if k != 'type'})

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
    path = os.path.join(os.getcwd(), filename)
    with open(path, 'r') as fp:
        config = json.load(fp=fp)
    return layered_network_builder(config)
