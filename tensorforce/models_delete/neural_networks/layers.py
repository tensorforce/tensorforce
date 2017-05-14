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

import copy

import tensorflow as tf
from tensorforce.util.config_util import make_function
import numpy as np

tf_slim = tf.contrib.slim


def layer_wrapper(layer_constructor, requires_episode_length=False, reshape=None):

    def layer_fn(layer_input, config, episode_length=None, scope=None):
        """
        Initialize layer of any kind.
        :param input: input layer
        :param config: layer configuration
        :param scope: tensorflow scope
        """
        kwargs = copy.copy(config)

        # Remove `type` from kwargs array.
        kwargs.pop("type", None)

        for fk in ["weights_initializer", "weights_regularizer", "biases_initializer", "activation_fn", "normalizer_fn"]:
            make_function(kwargs, fk)

        # Force our own scope definitions
        if scope:
            kwargs['scope'] = scope

        if reshape and callable(reshape):
            layer_input = reshape(layer_input)

        if requires_episode_length:
            assert episode_length is not None
            return layer_constructor(layer_input, episode_length, **kwargs)
        else:
            return layer_constructor(layer_input, **kwargs)

    return layer_fn


def flatten_layer(layer_input, scope=None, **kwargs):
    with tf.name_scope(scope or 'lstm'):
        size = 1
        for dim in layer_input.get_shape().as_list()[2:]:
            size *= dim
        flattened = tf.map_fn(fn=(lambda t: tf.reshape(tensor=t, shape=(-1, size))), elems=layer_input)
        # flattened = tf.reshape(tensor=layer_input, shape=(-1, layer_input.get_shape()[1].value, size))
    return flattened


def conv2d_layer(layer_input, scope=None, **kwargs):
    with tf.name_scope(scope or 'conv2d'):
        conv2d = tf.map_fn(fn=(lambda t: tf_slim.conv2d(t, **kwargs)), elems=layer_input)
    return conv2d

#TODO fully custom layer implementations are necessary to control internal state and shapes better

def lstm_layer(layer_input, episode_length, scope=None, **kwargs):  # lstm_size
    """
    Creates an LSTM layer.
    
    :param layer_input: 
    :param episode_length: 
    :param kwargs: 
    :return: 
    """
    with tf.name_scope(scope or 'lstm'):
        lstm_size = layer_input.get_shape()[2].value
        internal_state_input = tf.placeholder(dtype=tf.float32, shape=(2, lstm_size))
        internal_state_init = np.zeros(shape=(2, lstm_size))

        initial_c = tf.expand_dims(input=internal_state_input[0, :], axis=0)
        initial_c = tf.concat(values=(initial_c, tf.zeros_like(tensor=layer_input[1:, 0, :])), axis=0)
        initial_h = tf.expand_dims(input=internal_state_input[1, :], axis=0)
        initial_h = tf.concat(values=(initial_h, tf.zeros_like(tensor=layer_input[1:, 0, :])), axis=0)
        initial_state = tf.contrib.rnn.LSTMStateTuple(c=initial_c, h=initial_h)

        lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_size)
        outputs, internal_state = tf.nn.dynamic_rnn(cell=lstm, inputs=layer_input, sequence_length=episode_length, initial_state=initial_state)
        internal_state_output = tf.stack(values=(internal_state.c[-1, :], internal_state.h[-1, :]))

    return outputs, internal_state_input, internal_state_output, internal_state_init


flatten = layer_wrapper(flatten_layer)
dense = layer_wrapper(tf_slim.fully_connected)
conv2d = layer_wrapper(conv2d_layer)
linear = layer_wrapper(tf_slim.linear)
lstm = layer_wrapper(lstm_layer, requires_episode_length=True)


layer_classes = {
    'flatten': flatten,
    'dense': dense,
    'conv2d': conv2d,
    'linear': linear,
    'lstm': lstm
}
