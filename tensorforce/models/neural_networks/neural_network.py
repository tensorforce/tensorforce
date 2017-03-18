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
Creates neural networks from a configuration dict.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

import tensorflow as tf

from tensorforce.exceptions.tensorforce_exceptions import ConfigError
from tensorforce.models.neural_networks.layers import layer_classes

tf_slim = tf.contrib.slim


class NeuralNetwork(object):

    def __init__(self, define_network, inputs, scope='value_function'):
        """
        A neural network.

        :param inputs: TF input placeholders
        :param scope: TF scope
        :return: A TensorFlow network
        """

        with tf.variable_scope(scope):
            self.inputs = inputs
            self.output = define_network(inputs)
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    def get_inputs(self):
        return self.inputs

    def get_output(self):
        return self.output

    def get_variables(self):
        return self.variables

    @staticmethod
    def layered_network(layers):
        """
        Returns a function defining a layered neural network according to the given configuration.

        :param layers: Dict that describes a neural network layer-wise
        :return: A function defining a TensorFlow network
        """

        if not layers:
            raise ConfigError("Invalid configuration, missing layer specification.")

        def define_network(inputs):
            assert len(inputs) == 1  # layered network only has one input
            layer = inputs[0]
            type_counter = Counter()

            for layer_config in layers:
                layer_type = layer_config['type']
                type_counter[layer_type] += 1
                layer_name = "{type}{num}".format(type=layer_type, num=type_counter[layer_type])
                layer = layer_classes[layer_type](layer, layer_config, layer_name)

            return layer  # set output to last layer

        return define_network
