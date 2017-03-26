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
Standard Gaussian policy.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorforce.models.neural_networks.layers import linear
from tensorforce.models.policies import StochasticPolicy
from tensorforce.models.policies.gaussian import Gaussian


class GaussianPolicy(StochasticPolicy):

    def __init__(self, network, session, state, random, action_count=1, scope='policy'):
        with tf.variable_scope(scope):
            action_means = linear(network.output, {'num_outputs': action_count}, 'action_mu')
            # Random init for log standard deviations
            log_standard_devs_init = tf.Variable(0.01 * random.randn(1, action_count), dtype=tf.float32)
            action_log_stds = tf.tile(log_standard_devs_init, tf.stack((tf.shape(action_means)[0], 1)))

        super(GaussianPolicy, self).__init__(network, [action_means, action_log_stds], session, state, random, action_count)
        self.dist = Gaussian(random)

    def sample(self, state):
        sample = super(GaussianPolicy, self).sample(state)
        action_means, action_log_stds = sample

        action = action_means + np.exp(action_log_stds) * self.random.normal(size=action_log_stds.shape)

        # ravel from [[]] to []
        return action.ravel(), dict(policy_output=action_means.ravel(),
                                    policy_log_std=action_log_stds.ravel())

    def get_distribution(self):
        return self.dist

    def get_policy_variables(self):
        return dict(policy_output=self.policy_outputs[0],
                    policy_log_std=self.policy_outputs[1])
