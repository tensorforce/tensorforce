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
Categorial one hot policy, used for discrete policy gradients.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorforce.models.neural_networks.layers import linear, flatten
from tensorforce.models.policies.categorical import Categorical
from tensorforce.models.policies.stochastic_policy import StochasticPolicy


class CategoricalOneHotPolicy(StochasticPolicy):

    def __init__(self, network, session, state, random, action_count=1, scope='policy'):
        with tf.variable_scope(scope):
            action_layer = linear(layer_input=network.output, config={'num_outputs': action_count}, scope='outputs')
            action_layer = tf.reshape(action_layer, [-1, action_count])

            distribution = tf.nn.softmax(action_layer)
            sample = tf.multinomial(distribution, 1)

        super(CategoricalOneHotPolicy, self).__init__(network, [distribution, sample], session, state, random, action_count)
        self.dist = Categorical(random)

    def get_distribution(self):
        return self.dist

    def sample(self, state, sample=True):
        output_dist, output_sample = super(CategoricalOneHotPolicy, self).sample(state)

        output_dist = output_dist.ravel()
        if sample:
            # We currently use tf.multinomial for sampling, as np.random.multinomial has a precision of 1e-12 and raises
            # ValueErrors when sum(pvals) > 1.0. With tensorflow's precision of 1e-8, this might happen.
            action = np.flatnonzero(output_sample)
        else:
            action = int(np.argmax(output_dist))

        one_hot = np.zeros_like(output_dist)
        one_hot[action] = 1

        # We return a one hot vector and then extract the concrete action in the pg agent
        # This is so we can have the same tensor shapes for discrete vs continuous actions
        return one_hot, dict(policy_output=output_dist)

    def get_policy_variables(self):
        return dict(policy_output=self.policy_outputs[0])
