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


class VPGModel(PGModel):
    default_config = VPGModelConfig

    def __init__(self, config, scope):
        super(VPGModel, self).__init__(config, scope)

        self.create_training_operations()
        self.session.run(tf.global_variables_initializer())

    def create_training_operations(self):
        with tf.variable_scope("update"):
            self.log_probabilities = self.dist.log_prob(self.policy.get_policy_variables(), self.actions)

            # Concise: Get log likelihood of actions, weigh by advantages, compute gradient on that
            self.loss = -tf.reduce_mean(self.log_probabilities * self.advantage, name="loss_op")

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
        _, _, actions, batch_advantage, states, path_lengths = self.merge_episodes(batch)

        fetches = [self.optimize_op, self.log_probabilities, self.loss]
        fetches.extend(self.network.internal_state_outputs)

        feed_dict = {self.state: states, self.path_length: path_lengths, self.actions: actions, self.advantage: batch_advantage}
        feed_dict.update({internal_state: self.internal_states[n] for n, internal_state in enumerate(self.network.internal_state_inputs)})

        fetched = self.session.run(fetches, feed_dict)
        log_probs = fetched[1]
        loss = fetched[2]
        self.internal_states = fetched[3:]
        # print('log probs:' + str(log_probs))
        # print('loss:' + str(loss))
