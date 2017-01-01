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
Vanilla policy gradient implementation.
"""
from tensorforce.config import create_config
from tensorforce.neural_networks import NeuralNetwork
from tensorforce.updater import Model
import numpy as np
from tensorforce.util.experiment_util import global_seed
from tensorforce.util.exploration_util import exploration_mode
import tensorflow as tf


class VanillaPolicyGradient(Model):

    default_config = {
        'gamma': 0.99,
    }

    def __init__(self, config):
        super(VanillaPolicyGradient, self).__init__(config)
        self.config = create_config(config, default=self.default_config)
        self.action_count = self.config.actions
        self.tau = self.config.tau
        self.epsilon = self.config.epsilon
        self.gamma = self.config.gamma
        self.alpha = self.config.alpha
        self.batch_size = self.config.batch_size

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")
        self.input_feed = None

        self.actions = tf.placeholder(tf.float32, [None, self.action_count], name='actions')
        self.advantage = tf.placeholder(tf.float32, shape=[None])

        scope = '' if self.config.tf_scope is None else self.config.tf_scope + '-'
        self.hidden_layers = NeuralNetwork(self.config.network_layers, self.state,
                                           scope=scope + 'value_function')
        self.exploration = exploration_mode[self.config.exploration_mode]

    def get_action(self, state):
        pass

    def create_training_operations(self):
        pass

    def update(self, batch):
        pass


