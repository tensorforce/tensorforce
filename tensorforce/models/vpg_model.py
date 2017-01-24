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

        self.create_training_operations()
        self.session.run(tf.global_variables_initializer())

    def create_training_operations(self):
        with tf.variable_scope("update"):
            # If output 0, log NaN -> add epsilon to outputs for good measure?
            self.log_probabilities = self.dist.log_prob(self.policy.get_policy_variables(), self.actions)

            self.loss = -tf.reduce_sum(self.log_probabilities * self.advantage, name="loss_op")

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
                      self.advantage: batch_advantage}

        self.session.run(self.optimize_op, input_feed)
        #print('loss:' + str(loss))



