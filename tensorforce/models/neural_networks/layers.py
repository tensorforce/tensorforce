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


def layer_wrapper(layer_constructor, requires_path_length=False, reshape=None):

    def layer_fn(layer_input, config, path_length=None, scope=None):
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
        kwargs['scope'] = scope

        if reshape and callable(reshape):
            layer_input = reshape(layer_input)

        if requires_path_length:
            assert path_length is not None
            return layer_constructor(layer_input, path_length, **kwargs)
        else:
            return layer_constructor(layer_input, **kwargs)

    return layer_fn


def lstm_layer(layer_input, path_length, **kwargs):
    assert layer_input.get_shape().ndims == 2
    path_length = tf.pad(tensor=path_length, paddings=((0, 100 - tf.shape(input=path_length)[0]),))
    path_length = tf.reshape(tensor=path_length, shape=(100,))
    lstm_size = layer_input.get_shape()[1].value
    paths = tf.split(value=layer_input, num_or_size_splits=path_length)
    lengths = tf.unstack(value=path_length)
    max_length = tf.reduce_max(input_tensor=path_length)
    paths = [tf.pad(tensor=path, paddings=((0, max_length - length), (0, 0))) for path, length in zip(paths, lengths)]
    paths = tf.stack(values=paths)
    paths = tf.reshape(tensor=paths, shape=(-1, max_length, lstm_size))
    internal_state_input = tf.placeholder(dtype=tf.float32, shape=(2, lstm_size))
    lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_size)
    zero_state = tf.zeros_like(tensor=paths[1:, 0, :])
    initial_c = tf.concat(values=(tf.expand_dims(input=internal_state_input[0, :], axis=0), zero_state), axis=0)
    initial_h = tf.concat(values=(tf.expand_dims(input=internal_state_input[1, :], axis=0), zero_state), axis=0)
    initial_state = tf.contrib.rnn.LSTMStateTuple(c=initial_c, h=initial_h)
    outputs, internal_state = tf.nn.dynamic_rnn(cell=lstm, inputs=paths, sequence_length=path_length, initial_state=initial_state)
    masks = tf.sequence_mask(lengths=path_length, maxlen=tf.reduce_max(input_tensor=path_length))
    outputs = tf.boolean_mask(tensor=outputs, mask=masks)
    internal_state_output = tf.stack(values=(internal_state.c[-1, :], internal_state.h[-1, :]))
    internal_state_init = np.zeros(shape=(2, lstm_size))
    return outputs, internal_state_input, internal_state_output, internal_state_init


dense = layer_wrapper(tf_slim.fully_connected)
conv2d = layer_wrapper(tf_slim.conv2d)
linear = layer_wrapper(tf_slim.linear, reshape=lambda input: tf.reshape(input, (-1, int(np.prod(input.get_shape()[1:])))))
lstm = layer_wrapper(lstm_layer, requires_path_length=True)


layer_classes = {
    'dense': dense,
    'conv2d': conv2d,
    'linear': linear,
    'lstm': lstm
}
