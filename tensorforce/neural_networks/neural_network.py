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
# =====

import tensorflow as tf
from six.moves import xrange

from tensorforce.neural_networks.layers import layers

tf_slim = tf.contrib.slim


def get_network(config, scope='value_function'):
    """
    Creates a neural network according to the given config.

    :param config: Describes a neural network layer wise
    :param scope: TF scope
    :return: A TensorFlow network
    """

    with tf.variable_scope(scope, [config['input_shape']]) as sc:

        type_counter = {}

        first_layer = True
        input = config['input']  # for the first layer
        for layer in config['layers']:
            type = layer['type']

            if first_layer:
                name = 'input'
                first_layer = False
            else:
                type_count = type_counter.get(type, 0)
                name = "{type}{num}".format(type=type, num=type_count+1)
                type_counter.update({type: type_count+1})

            network = layers[type](input, layer, name)
            input = network  # for all subsequent layers

    return network