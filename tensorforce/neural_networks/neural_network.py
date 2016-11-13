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
Creates neural networks from a configuration dict.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorforce.exceptions.tensorforce_exceptions import ConfigError
from tensorforce.neural_networks.layers import layers

tf_slim = tf.contrib.slim


def get_layers(network_layers, input_data, scope='value_function'):
    """
    Creates a neural network according to the given config.

    :param network_layers: Dict that describes a neural network layer wise
    :param input_data: TF input placeholder
    :param scope: TF scope
    :return: A TensorFlow network
    """

    with tf.variable_scope(scope):

        type_counter = {}

        if not network_layers:
            raise ConfigError("Invalid configuration, missing layer specification.")

        first_layer = True
        layer = input_data  # for the first layer

        for layer_config in network_layers:
            layer_type = layer_config['type']

            if first_layer:
                name = 'input'
                first_layer = False
            else:
                type_count = type_counter.get(layer_type, 0)
                name = "{type}{num}".format(type=layer_type, num=type_count + 1)
                type_counter.update({layer_type: type_count + 1})

            network = layers[layer_type](layer, layer_config, name)

        return layer
