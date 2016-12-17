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
Implements trust region policy optimization with general advantage estimation (TRPO-GAE) as
introduced by Schulman et al.

Largely based on https://github.com/ilyasu123/trpo, with a hopefully more readable
modularisation.

"""
from tensorforce.config import create_config
from tensorforce.neural_networks.layers import dense
from tensorforce.neural_networks.neural_network import NeuralNetwork
from tensorforce.updater.value_function import ValueFunction
from tensorforce.util.experiment_util import global_seed
from tensorforce.util.math_util import get_log_prob_gaussian
import numpy as np
import tensorflow as tf



class TRPOUpdater(ValueFunction):
    default_config = {
        'cg_damping': 0.1,
        'max_kl_divergence': 0.01,
    }

    def __init__(self, config):
        super(TRPOUpdater, self).__init__(config)

        self.config = create_config(config, default=self.default_config)
        self.batch_size = config.batch_size
        self.action_count = self.config.actions
        self.gamma = self.config.gamma

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        if self.config.concat is not None and self.config.concat > 1:
            self.state = tf.placeholder(tf.float32, [None, self.config.concat_length] + list(self.config.state_shape),
                                        name="state")
        else:
            self.state = tf.placeholder(tf.float32, [None] + list(self.config.state_shape), name="state")

        self.episode = 0

        self.actions = tf.placeholder(tf.float32, [None, self.action_count], name='actions')
        self.advantage = tf.placeholder(tf.float32, shape=[None])
        self.hidden_layers = NeuralNetwork(self.config.network_layers, self.state, 'training')

        self.saver = tf.train.Saver()
        self.variables = tf.trainable_variables()
        self.session.run(tf.initialize_all_variables())

    def create_outputs(self):
        # Output action means and log standard deviations
        with tf.variable_scope("policy"):
            self.action_means = dense(self.hidden_layers.get_output(),
                                     {'neurons': self.action_count, 'regularization': self.config.regularizer,
                                      'regularization_param': self.config.regularization_param}, 'action_mu')
            self.prev_action_means = tf.placeholder(tf.float32, [None, self.action_count])

            # Random init for log standard deviations
            log_standard_devs_init = tf.Variable(0.01 * self.random.randn(1, self.action_count))

            self.action_log_stds = tf.tile(log_standard_devs_init, tf.pack((tf.shape(self.action_means)[0], 1)))
            self.prev_action_log_stds = tf.placeholder(tf.float32, [None, self.action_count])


    def create_training_operations(self):
        """
        Creates TRPO training operations, i.e. the natural gradient update step
        based on the KL divergence constraint between new and old policy.
        :return:
        """
        with tf.variable_scope("update"):
            current_log_prob  = get_log_prob_gaussian(self.action_means, self.action_log_stds, self.actions)
            prev_log_prob = get_log_prob_gaussian(self.prev_action_means, self.prev_action_log_stds, self.actions)

            prob_ratio = tf.exp(current_log_prob - prev_log_prob)

            surrogate = -tf.reduce_mean(prob_ratio * self.advantage)




    def get_action(self, state):
        pass

    def update(self, batch):
        """
        Compute update for one batch of experiences.
        :param batch:
        :return:
        """

        pass

    def calculate_advantage(self, batch):
        # Estimate advantage for given batch
        pass
