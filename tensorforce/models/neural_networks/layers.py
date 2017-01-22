# Copyright 2016 reinforce.io. All Rights Reserved.
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

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorforce.util.config_util import get_function
import numpy as np

tf_slim = tf.contrib.slim


def dense(input, config, scope):
    """
    Fully connected layer.

    :param input: Input to the layer, e.g. handle to another layer
    :param config: Layer config
    :param scope: Layer name
    :return:
    """
    kwargs = {
        'weights_initializer': get_function(config.get('weight_init'),
                                            config.get('weight_init_param'),
                                            initializers.xavier_initializer()),
        'biases_initializer': get_function(config.get('bias_init'),
                                           config.get('bias_init_param'),
                                           init_ops.zeros_initializer),
        'activation_fn': get_function(config.get('activation'),
                                      config.get('activation_param'),
                                      nn.relu),
        'weights_regularizer': get_function(config.get('regularization'),
                                            config.get('regularization_param'),
                                            None),
        'normalizer_fn': get_function(config.get('normalizer'),
                                            config.get('normalizer_param'),
                                            None)
    }
    # Flatten

    return tf_slim.fully_connected(input,
                                   config['neurons'],
                                   scope=scope,
                                   **kwargs)


def conv2d(input, config, scope):
    """
    Convolutional 2d layer.

    :param input: Input to the layer, e.g. handle to another layer
    :param config: Layer config
    :param scope: Layer name
    :return:
    """
    kwargs = {
        'weights_initializer': get_function(config.get('weight_init'),
                                            config.get('weight_init_param'),
                                            initializers.xavier_initializer()),
        'biases_initializer': get_function(config.get('bias_init'),
                                           config.get('bias_init_param'),
                                           init_ops.zeros_initializer),
        'activation_fn': get_function(config.get('activation'),
                                      config.get('activation_param'),
                                      nn.relu),
        'weights_regularizer': get_function(config.get('regularization'),
                                            config.get('regularization_param'),
                                            None),
        'padding': config.get('padding', 'VALID')
    }

    return tf_slim.conv2d(input,
                          config['neurons'],
                          config['kernel_size'],
                          config['stride'],
                          scope=scope,
                          **kwargs)


def linear(input, config, scope):
    """
    Fully connected layer.

    :param input: Input to the layer, e.g. handle to another layer
    :param config: Layer config
    :param scope: Layer name
    :return:
    """

    kwargs = {
        'weights_initializer': get_function(config.get('weight_init'),
                                            config.get('weight_init_param'),
                                            initializers.xavier_initializer()),
        'biases_initializer': get_function(config.get('bias_init'),
                                           config.get('bias_init_param'),
                                           init_ops.zeros_initializer),
        'weights_regularizer': get_function(config.get('regularization'),
                                            config.get('regularization_param'),
                                            None)
    }
    # Flatten
    input = tf.reshape(input, (-1, int(np.prod(input.get_shape()[1:]))))

    return tf_slim.linear(input,
                          config['neurons'],
                          scope=scope,
                          **kwargs)

layers = {
    'dense': dense,
    'conv2d': conv2d,
    'linear': linear
}
