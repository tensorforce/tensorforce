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

    def __init__(self, config):
        config.default(VPGModel.default_config)
        super(VPGModel, self).__init__(config)

    def create_tf_operations(self, config):
        super(VPGModel, self).create_tf_operations(config)

        with tf.variable_scope('update'):
            for name, action in self.action.items():
                log_prob = self.distribution[name].log_probability(action=action)
                loss = -tf.reduce_mean(input_tensor=tf.multiply(x=log_prob, y=self.reward), axis=0)
                tf.losses.add_loss(loss)
