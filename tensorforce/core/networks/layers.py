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
import copy
import json
from math import sqrt
import numpy as np
import os
import tensorflow as tf

from tensorforce import util

tf_slim = tf.contrib.slim

# TODO configurable initialisation

def layer_wrapper(layer_constructor, requires_episode_length=False, reshape=None):

    def layer(x, config, episode_length=None, scope=None):
        """
        Initialises layer ased on layer config. 
        Args:
            x: 
            config: 
            episode_length: 
            scope: 

        Returns:

        """
        kwargs = copy.copy(config)

        # Remove `type` from kwargs array.
        kwargs.pop("type", None)

        # for fk in ["weights_initializer", "weights_regularizer", "biases_initializer", "activation_fn", "normalizer_fn"]:
        #     util.make_function(kwargs, fk)

        # Force our own scope definitions
        if scope:
            kwargs['scope'] = scope

        if reshape and callable(reshape):
            x = reshape(x)

        if requires_episode_length:
            assert episode_length is not None
            return layer_constructor(x, episode_length, **kwargs)
        else:
            return layer_constructor(x, **kwargs)

    return layer


def flatten_layer(x):
    with tf.variable_scope('flatten'):
        x = tf.reshape(tensor=x, shape=(-1, util.prod(x.get_shape().as_list()[1:])))
    return x


def linear_layer(x, size, l2_regularization=0.0):
    with tf.variable_scope('linear'):
        weights = tf.Variable(initial_value=tf.random_normal(shape=(x.get_shape()[1].value, size), stddev=sqrt(2.0 / (x.get_shape()[1].value + size))))
        bias = tf.Variable(initial_value=tf.zeros(shape=(size,)))
        if l2_regularization > 0.0:
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=weights))
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=bias))
        x = tf.nn.bias_add(value=tf.matmul(a=x, b=weights), bias=bias)
    return x


def dense_layer(x, size, l2_regularization=0.0):
    with tf.variable_scope('dense'):
        weights = tf.Variable(initial_value=tf.random_normal(shape=(x.get_shape()[1].value, size), stddev=sqrt(2.0 / (x.get_shape()[1].value + size))))
        bias = tf.Variable(initial_value=tf.zeros(shape=(size,)))
        if l2_regularization > 0.0:
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=weights))
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=bias))
        x = tf.nn.bias_add(value=tf.matmul(a=x, b=weights), bias=bias)
        x = tf.nn.relu(features=x)
    return x


def conv2d_layer(x, size, window=3, stride=1, l2_regularization=0.0):
    with tf.variable_scope('conv2d'):
        filters = tf.Variable(initial_value=tf.random_normal(shape=(window, window, x.get_shape()[3].value, size), stddev=sqrt(2.0 / size)))
        if l2_regularization > 0.0:
            tf.losses.add_loss(l2_regularization * tf.nn.l2_loss(t=filters))
        x = tf.nn.conv2d(input=x, filter=filters, strides=(1, stride, stride, 1), padding='SAME')
        x = tf.nn.relu(features=x)
    return x


def lstm_layer(x, size=None, l2_regularization=0.0):
    """
    Creates an LSTM layer.
    """
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
    'flatten': flatten_layer,
    'dense': dense_layer,
    'conv2d': conv2d_layer,
    'linear': linear_layer,
    'lstm': lstm_layer
}


def layered_network_builder(layers_config):
    """
    Returns a function defining a layered neural network according to the given configuration.

    :param layers: Dict that describes a neural network layer-wise
    :return: A function defining a TensorFlow network
    """

    def network_builder(inputs):
        if len(inputs) != 1:  # layered network only has one input
            raise Exception()
        layer = next(iter(inputs.values()))
        internal_inputs = []
        internal_outputs = []
        internal_inits = []

        type_counter = Counter()
        for layer_config in layers_config:
            layer_type = layer_config['type']
            type_counter[layer_type] += 1
            layer = layers[layer_type](x=layer, **{key: value for key, value in layer_config.items() if key != 'type'})

            if isinstance(layer, list) or isinstance(layer, tuple):
                assert len(layer) == 4
                internal_inputs.extend(layer[1])
                internal_outputs.extend(layer[2])
                internal_inits.extend(layer[3])
                layer = layer[0]

        if internal_inputs:
            return layer, internal_inputs, internal_outputs, internal_inits
        else:
            return layer

    return network_builder


def from_json(filename):
    path = os.path.join(os.getcwd(), filename)
    with open(path, 'r') as fp:
        config = json.load(fp=fp)
    return layered_network_builder(config)
