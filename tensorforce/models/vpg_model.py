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

from tensorforce.core import PolicyGradientModel


class VPGModel(PolicyGradientModel):

    allows_discrete_actions = True
    allows_continuous_actions = True
    default_config = dict()

    def __init__(self, config, network_builder):
        config.default(VPGModel.default_config)
        super(VPGModel, self).__init__(config, network_builder)

    def create_tf_operations(self, config):
        super(VPGModel, self).create_tf_operations(config)

        with tf.variable_scope('update'):
            for name, action in self.action.items():
                log_prob = self.distribution[name].log_probability(action=action)
                loss = -tf.reduce_mean(input_tensor=tf.multiply(x=log_prob, y=self.reward), axis=0)
                tf.losses.add_loss(loss)

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the vanilla policy gradient.
        :param batch:
        :return:
        """
        super(VPGModel, self).update(batch)

        fetches = [self.optimize, self.loss]

        feed_dict = {self.state[name]: batch['states'][name] for name in self.state}
        feed_dict.update({self.action[name]: batch['actions'][name] for name in self.action})
        feed_dict[self.reward] = batch['advantages']
        feed_dict.update({internal: batch['internals'][n] for n, internal in enumerate(self.network.internal_inputs)})

        _, loss = self.session.run(fetches=fetches, feed_dict=feed_dict)

        if self.logger:
            self.logger.debug('VPG loss = ' + str(loss))
