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


tf_slim = tf.contrib.slim


def dense(input_handle, config, scope):
    """
    Fully connected layer.

    :param input_handle: Input to the layer, e.g. handle to another layer
    :param config: Layer config
    :param scope: Layer name
    :return:
    """
    kwargs = {
        'weights_initializer':  get_function(config.get('weight_init'),
                                             config.get('weight_init_param'),
                                             initializers.xavier_initializer()),
        'biases_initializer':   get_function(config.get('bias_init'),
                                             config.get('bias_init_param'),
                                             init_ops.zeros_initializer),
        'activation_fn':        get_function(config.get('activation'),
                                             config.get('activation_param'),
                                             nn.relu),
        'weights_regularizer':  get_function(config.get('regularization'),
                                             config.get('regularization_param'),
                                             None),
    }

    return tf_slim.fully_connected(input_handle,
                                   config['neurons'],
                                   scope=scope,
                                   **kwargs)


def conv2d(input_handle, config, scope):
    """
    Convolutional 2d layer.

    :param input_handle: Input to the layer, e.g. handle to another layer
    :param config: Layer config
    :param scope: Layer name
    :return:
    """
    kwargs = {
        'weights_initializer':  get_function(config.get('weight_init'),
                                             config.get('weight_init_param'),
                                             initializers.xavier_initializer()),
        'biases_initializer':   get_function(config.get('bias_init'),
                                             config.get('bias_init_param'),
                                             init_ops.zeros_initializer),
        'activation_fn':        get_function(config.get('activation'),
                                             config.get('activation_param'),
                                             nn.relu),
        'weights_regularizer':  get_function(config.get('regularization'),
                                             config.get('regularization_param'),
                                             None),
    }
    return tf_slim.conv2d(input_handle,
                          config['neurons'],
                          config['kernel_size'],
                          config['stride'],
                          scope=scope,
                          **kwargs)


layers = {
    'dense': dense,
    'conv2d': conv2d
}
