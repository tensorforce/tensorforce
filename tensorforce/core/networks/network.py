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

import tensorflow as tf


class NeuralNetwork(object):

    def __init__(self, network_builder, inputs):
        """
        A neural network.

        :param inputs: TF input placeholders
        :return: A TensorFlow network
        """
        network = network_builder(inputs)
        if isinstance(network, tf.Tensor):
            self.output = network
            self.internal_inputs = []
            self.internal_outputs = []
            self.internal_inits = []
        else:
            assert len(network) == 4
            self.output = network[0]
            self.internal_inputs = network[1]
            self.internal_outputs = network[2]
            self.internal_inits = network[3]
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
