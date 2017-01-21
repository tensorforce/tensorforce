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
import numpy as np
import tensorflow as tf

from tensorforce.models import LinearValueFunction
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.models.neural_networks.layers import linear
from tensorforce.models.pg_model import PGModel
from tensorforce.util.experiment_util import global_seed


class VPGModel(PGModel):
    default_config = {
        'gamma': 0.99,
        'use_gae' : False,
        'gae_lambda': 0.97  # GAE-lambda
    }

    def __init__(self, config):
        super(VPGModel, self).__init__(config)
        self.action_count = self.config.actions
        self.gamma = self.config.gamma
        self.batch_size = self.config.batch_size
        self.gae_lambda = self.config.gae_lambda
        self.use_gae = self.config.use_gae

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")
        self.episode = 0
        self.input_feed = None

        self.actions = tf.placeholder(tf.float32, [None, self.action_count], name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=[None])
        self.prev_action_means = tf.placeholder(tf.float32, [None, self.action_count])
        self.prev_action_log_stds = tf.placeholder(tf.float32, [None, self.action_count])

        scope = '' if self.config.tf_scope is None else self.config.tf_scope + '-'
        self.hidden_layers = NeuralNetwork(self.config.network_layers, self.state,
                                           scope=scope + 'value_function')

        self.saver = tf.train.Saver()
        self.create_outputs()
        self.baseline_value_function = LinearValueFunction()
        self.create_training_operations()

        self.session.run(tf.global_variables_initializer())

    def create_training_operations(self):
        with tf.variable_scope("update"):
            log_probabilities = tf.log(tf.reduce_sum(tf.mul(self.outputs, self.actions), reduction_indices=[1]))

            self.loss = -tf.reduce_mean(log_probabilities * self.advantages, name="loss_op")

            self.optimize_op = self.optimizer.minimize(self.loss)

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the vanilla policy gradient.
        :param batch:
        :return:
        """
        # Set per episode advantage using GAE
        self.compute_gae_advantage(batch, self.gamma, self.gae_lambda)

        # Update linear value function for baseline prediction
        self.baseline_value_function.fit(batch)

        # Merge episode inputs into single arrays
        _, _, actions, batch_advantage, states = self.merge_episodes(batch)

        input_feed = {self.state: states,
                      self.actions: actions,
                      self.advantages: batch_advantage}

        self.session.run(self.optimize_op, input_feed)

    def create_outputs(self):
        with tf.variable_scope("policy"):
            # TODO softmax here is meant for discrete actions, needs to be an optional
            # TODO mapping to e.g. a Gaussian for continuous actions
            self.outputs = linear(self.hidden_layers.get_output(),
                                 {'neurons': self.action_count, 'regularization': self.config.regularizer,
                                  'regularization_param': self.config.regularization_param,
                                  'activation': tf.nn.softmax}, 'action_mu')
            # Random init for log standard deviations
            log_standard_devs_init = tf.Variable(0.01 * self.random.randn(1, self.action_count), dtype=tf.float32)

            self.action_log_stds = tf.tile(log_standard_devs_init, tf.pack((tf.shape(self.outputs)[0], 1)))
