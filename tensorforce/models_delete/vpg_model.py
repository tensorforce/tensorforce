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
Vanilla policy gradient implementation.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce.models.pg_model import PGModel
from tensorforce.default_configs import VPGModelConfig
from tensorforce.util.math_util import discount


class VPGModel(PGModel):
    default_config = VPGModelConfig

    def __init__(self, config, scope, network_builder=None):
        super(VPGModel, self).__init__(config, scope, network_builder=network_builder)

        self.create_training_operations()
        self.session.run(tf.global_variables_initializer())

    def create_training_operations(self):
        with tf.variable_scope("update"):
            self.log_probabilities = self.dist.log_prob(self.policy.get_policy_variables(), self.actions)

            self.log_probabilities = tf.reshape(self.log_probabilities, [-1])

            # Concise: Get log likelihood of actions, weigh by advantages, compute gradient on that
            self.loss = -tf.reduce_mean(self.log_probabilities * tf.reshape(self.advantage, [-1]), name="loss_op", axis=0)

            self.optimize_op = self.optimizer.minimize(self.loss)

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the vanilla policy gradient.
        :param batch:
        :return:
        """

        # Set per episode return and advantage
        for episode in batch:
            episode['returns'] = discount(episode['rewards'], self.gamma)
            episode['advantages'] = self.advantage_estimation(episode)

        # Update linear value function for baseline prediction
        self.baseline_value_function.fit(batch)

        fetches = [self.optimize_op, self.log_probabilities, self.loss]
        fetches.extend(self.network.internal_state_outputs)

        feed_dict = {
            self.episode_length: [episode['episode_length'] for episode in batch],
            self.state: [episode['states'] for episode in batch],
            self.actions: [episode['actions'] for episode in batch],
            self.advantage: [episode['advantages'] for episode in batch]
        }

        for n, internal_state in enumerate(self.network.internal_state_inputs):
            feed_dict[internal_state] = self.internal_states[n]

        fetched = self.session.run(fetches, feed_dict)

        loss = fetched[2]
        self.internal_states = fetched[3:]

        self.logger.debug('Vanilla policy gradient loss = ' + str(loss))
