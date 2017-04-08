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
Implements trust region policy optimization with general advantage estimation (TRPO-GAE) as
introduced by Schulman et al.

Based on https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py and
https://github.com/ilyasu123/trpo, with a hopefully slightly more readable
modularisation and some modifications.

The core training update code is under MIT license, for more information see LICENSE-EXT.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorforce.models.pg_model import PGModel
from tensorforce.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from tensorforce.util.math_util import *
from tensorforce.default_configs import TRPOModelConfig


class TRPOModel(PGModel):
    default_config = TRPOModelConfig

    def __init__(self, config, scope):
        super(TRPOModel, self).__init__(config, scope)

        # TRPO specific parameters
        self.cg_damping = self.config.cg_damping
        self.max_kl_divergence = self.config.max_kl_divergence
        self.line_search_steps = self.config.line_search_steps
        self.cg_optimizer = ConjugateGradientOptimizer(self.logger, self.config.cg_iterations)

        self.flat_tangent = tf.placeholder(tf.float32, shape=[None])
        self.create_training_operations()
        self.session.run(tf.global_variables_initializer())

    def create_training_operations(self):
        """
        Creates TRPO training operations, i.e. the natural gradient update step
        based on the KL divergence constraint between new and old policy.
        :return:
        """

        with tf.variable_scope("update"):
            current_log_prob = self.dist.log_prob(self.policy.get_policy_variables(), self.actions)
            prev_log_prob = self.dist.log_prob(self.prev_dist, self.actions)

            prob_ratio = tf.exp(current_log_prob - prev_log_prob)
            surrogate_loss = -tf.reduce_mean(prob_ratio * self.advantage)
            variables = tf.trainable_variables()
            batch_float = tf.cast(self.batch_size, tf.float32)

            mean_kl_divergence = self.dist.kl_divergence(self.prev_dist, self.policy.get_policy_variables())\
                                 / batch_float
            mean_entropy = self.dist.entropy(self.policy.get_policy_variables()) / batch_float

            self.losses = [surrogate_loss, mean_kl_divergence, mean_entropy]

            # Get symbolic gradient expressions
            self.policy_gradient = get_flattened_gradient(self.losses, variables)
            fixed_kl_divergence = self.dist.fixed_kl(self.policy.get_policy_variables()) / batch_float

            variable_shapes = map(get_shape, variables)
            offset = 0
            tangents = []
            for shape in variable_shapes:
                size = np.prod(shape)
                param = tf.reshape(self.flat_tangent[offset:(offset + size)], shape)
                tangents.append(param)
                offset += size

            gradients = tf.gradients(fixed_kl_divergence, variables)
            gradient_vector_product = [tf.reduce_sum(g * t) for (g, t) in zip(gradients, tangents)]

            self.flat_variable_helper = FlatVarHelper(self.session, variables)
            self.fisher_vector_product = get_flattened_gradient(gradient_vector_product, variables)

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the constrained optimisation based on the fixed kl-divergence constraint.

        :param batch:
        :return:
        """
        self.input_feed = None

        # Set per episode return and advantage
        for episode in batch:
            episode['returns'] = discount(episode['rewards'], self.gamma)
            episode['advantages'] = self.generalised_advantage_estimation(episode)

        # Update linear value function for baseline prediction
        self.baseline_value_function.fit(batch)

        self.input_feed = {
            self.episode_length: [episode['episode_length'] for episode in batch],
            self.state: [episode['states'] for episode in batch],
            self.actions: [episode['actions'] for episode in batch],
            self.advantage: [episode['advantages'] for episode in batch],
            self.prev_action_means: [episode['action_means'] for episode in batch]
        }

        if self.continuous:
            self.input_feed[self.prev_action_log_stds] = [episode['action_log_stds']
                                                          for episode in batch]

        previous_theta = self.flat_variable_helper.get()

        gradient = self.session.run(self.policy_gradient, self.input_feed)
        zero = np.zeros_like(gradient)

        if np.allclose(gradient, zero):
            self.logger.debug('Gradient zero, skipping update')
        else:
            # The details of the approximations used here to solve the constrained
            # optimisation can be found in Appendix C of the TRPO paper
            # Note that no subsampling is used, which would improve computational performance
            search_direction = self.cg_optimizer.solve(self.compute_fvp, -gradient)

            # Search direction has now been approximated as cg-solution s= A^-1g where A is
            # Fisher matrix, which is a local approximation of the
            # KL divergence constraint
            shs = 0.5 * search_direction.dot(self.compute_fvp(search_direction))
            lagrange_multiplier = np.sqrt(shs / self.max_kl_divergence)
            update_step = search_direction / lagrange_multiplier
            negative_gradient_direction = -gradient.dot(search_direction)

            # Improve update step through simple backtracking line search
            # N.b. some implementations skip the line search
            improved, theta = line_search(self.compute_surrogate_loss, previous_theta, update_step,
                                          negative_gradient_direction / lagrange_multiplier, self.line_search_steps)

            # Use line search results, otherwise take full step
            # N.B. some implementations don't use the line search
            if improved:
                self.logger.debug('Updating with line search result..')
                self.flat_variable_helper.set(theta)
            else:
                self.logger.debug('Updating with full step..')
                self.flat_variable_helper.set(previous_theta + update_step)

            # Get loss values for progress monitoring
            # Optionally manage internal LSTM state or other relevant state

            for n, internal_state in enumerate(self.network.internal_state_inputs):
                self.input_feed[internal_state] = self.internal_states[n]

            self.losses.extend(self.network.internal_state_outputs)
            fetched = self.session.run(self.losses, self.input_feed)

            # Sanity checks. Is entropy decreasing? Is KL divergence within reason? Is loss non-zero?
            self.logger.info('Surrogate loss = ' + str(fetched[0]))
            self.logger.info('KL-divergence after update = ' + str(fetched[1]))
            self.logger.info('Entropy = ' + str(fetched[2]))

            # Update internal state optionally
            self.internal_states = fetched[3:]

    def compute_fvp(self, p):
        self.input_feed[self.flat_tangent] = p

        return self.session.run(self.fisher_vector_product, self.input_feed) + p * self.cg_damping

    def compute_surrogate_loss(self, theta):
        self.flat_variable_helper.set(theta)

        # Losses[0] = surrogate_loss
        return self.session.run(self.losses[0], self.input_feed)
