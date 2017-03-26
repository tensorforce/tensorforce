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
from tensorforce.models.neural_networks.layers import linear
from tensorforce.models.policies.categorical import Categorical
from tensorforce.models.policies.stochastic_policy import StochasticPolicy
import numpy as np
import tensorflow as tf


class CategoricalOneHotPolicy(StochasticPolicy):

    def __init__(self, network, session, state, random, action_count=1, scope='policy'):
        with tf.variable_scope(scope):
            action_layer = linear(layer_input=network.output, config={'num_outputs': action_count}, scope='outputs')
            policy_output = tf.nn.softmax(action_layer)

        super(CategoricalOneHotPolicy, self).__init__(network, [policy_output], session, state, random, action_count)
        self.dist = Categorical(random)

    def get_distribution(self):
        return self.dist

    def sample(self, state, sample=True):
        sample = super(CategoricalOneHotPolicy, self).sample(state)
        output_dist = sample[0]

        output_dist = output_dist.ravel()
        if sample:
            action = self.dist.sample(dict(policy_output=output_dist))
        else:
            action = int(np.argmax(output_dist))

        one_hot = np.zeros_like(output_dist)
        one_hot[action] = 1

        # We return a one hot vector and then extract the concrete action in the pg agent
        # This is so we can have the same tensor shapes for discrete vs continuous actions
        return one_hot, dict(policy_output=output_dist)

    def get_policy_variables(self):
        return dict(policy_output=self.policy_outputs[0])
