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
A policy gradient agent provides generic methods used in pg algorithms, e.g.
GAE-computation or merging of episode data.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError, util
from tensorforce.core import Model
from tensorforce.core.networks import NeuralNetwork
from tensorforce.core.value_functions import value_functions
from tensorforce.core.distributions import distributions


class PolicyGradientModel(Model):

    default_config = dict(
        sample_actions=True,
        baseline=None,
        baseline_args=None,
        baseline_kwargs=None,
        generalized_advantage_estimation=False,
        gae_lambda=0.97,
        normalize_advantage=False
    )

    def __init__(self, config):
        config.default(PolicyGradientModel.default_config)

        # distribution
        self.distribution = dict()
        for name, action in config.actions:
            if 'distribution' in action:
                distribution = action.distribution
            else:
                distribution = 'gaussian' if action.continuous else 'categorical'
            if distribution not in distributions:
                raise TensorForceError()
            if action.continuous:
                self.distribution[name] = distributions[distribution]()
            else:
                self.distribution[name] = distributions[distribution](num_actions=action['num_actions'])

        # baseline
        baseline = config.baseline
        args = config.baseline_args or ()
        kwargs = config.baseline_kwargs or {}
        if config.baseline is None:
            self.baseline = None
        elif config.baseline in value_functions:
            self.baseline = value_functions[baseline](self.session, *args, **kwargs)
        else:
            raise Exception()

        super(PolicyGradientModel, self).__init__(config)

        # advantage estimation
        self.generalized_advantage_estimation = config.generalized_advantage_estimation
        if self.generalized_advantage_estimation:
            self.gae_lambda = config.gae_lambda
        self.normalize_advantage = config.normalize_advantage

    def create_tf_operations(self, config):
        super(PolicyGradientModel, self).create_tf_operations(config)

        with tf.variable_scope('value_function'):
            self.network = NeuralNetwork(config.network, inputs=self.state)
            self.internal_inputs.extend(self.network.internal_inputs)
            self.internal_outputs.extend(self.network.internal_outputs)
            self.internal_inits.extend(self.network.internal_inits)

        with tf.variable_scope('distribution'):
            for action, distribution in self.distribution.items():
                distribution.create_tf_operations(x=self.network.output, sample=config.sample_actions)
                self.action_taken[action] = distribution.value

        if self.baseline:
            with tf.variable_scope('baseline'):
                self.baseline.create_tf_operations(config)

    def update(self, batch):
        """Generic policy gradient update on a batch of experiences. Each model needs to update its specific
        logic.
        
        Args:
            batch: 

        Returns:

        """
        batch['returns'] = util.cumulative_discount(rewards=batch['rewards'], terminals=batch['terminals'], discount=self.discount)
        # assert utils.discount(batch['rewards'], batch['terminals'], self.discount) == discount
        batch['rewards'] = self.advantage_estimation(batch)
        if self.baseline:
            self.baseline.update(states=batch['states'], returns=batch['returns'])
        super(PolicyGradientModel, self).update(batch)

    def advantage_estimation(self, batch):
        """Expects a batch, returns advantages according to config.

        Args:
            batch: 

        Returns:

        """
        if not self.baseline:
            return batch['returns']

        estimates = self.baseline.predict(states=batch['states'])
        if self.generalized_advantage_estimation:
            deltas = np.array(self.discount * estimates[n + 1] - estimates[n] if n < len(estimates) - 1 or terminal else 0.0 for n, terminal in enumerate(batch['terminals']))
            deltas += batch['rewards']
            # if terminals[-1]:
            #     adjusted_estimate = np.append(estimate, [0])
            # else:
            #     adjusted_estimate = np.append(estimate, estimate[-1])
            # deltas = batch['rewards'] + self.discount * adjusted_estimate[1:] - adjusted_estimate[:-1]
            advantage = util.cumulative_discount(rewards=deltas, terminals=batch['terminals'], discount=(self.discount * self.gae_lambda))
        else:
            advantage = batch['returns'] - estimates

        if self.normalize_advantage:
            advantage -= advantage.mean()
            advantage /= advantage.std() + 1e-8

        return advantage
