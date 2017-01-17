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
Implements trust region policy optimization with general advantage estimation (TRPO-GAE) as
introduced by Schulman et al.

Based on https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py and
https://github.com/ilyasu123/trpo, with a hopefully slightly more readable
modularisation and some modifications.

"""
from tensorforce.config import create_config
from tensorforce.neural_networks.layers import dense, linear
from tensorforce.neural_networks import NeuralNetwork
from tensorforce.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from tensorforce.updater import LinearValueFunction
from tensorforce.updater import Model
from tensorforce.updater.pg_model import PGModel
from tensorforce.util.experiment_util import global_seed
from tensorforce.util.math_util import *

import numpy as np
import tensorflow as tf


class TRPOUpdater(PGModel):
    default_config = {
        'cg_damping': 0.01,
        'cg_iterations': 15,
        'max_kl_divergence': 0.01,
        'gamma': 0.99,
        'use_gae': False,
        'gae_lambda': 0.97,  # GAE-lambda
        'line_search_steps': 10
    }

    def __init__(self, config):
        super(TRPOUpdater, self).__init__(config)
        self.batch_size = self.config.batch_size
        self.action_count = self.config.actions
        self.cg_damping = self.config.cg_damping
        self.line_search_steps = self.config.line_search_steps
        self.max_kl_divergence = self.config.max_kl_divergence
        self.use_gae = self.config.use_gae
        self.gae_lambda = self.config.gae_lambda
        self.cg_optimizer = ConjugateGradientOptimizer(self.config.cg_iterations)

        self.gamma = self.config.gamma

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")
        self.episode = 0
        self.input_feed = None

        self.actions = tf.placeholder(tf.float32, [None, self.action_count], name='actions')
        self.advantage = tf.placeholder(tf.float32, shape=[None])
        self.flat_tangent = tf.placeholder(tf.float32, shape=[None])
        self.prev_action_means = tf.placeholder(tf.float32, [None, self.action_count])
        self.prev_action_log_stds = tf.placeholder(tf.float32, [None, self.action_count])

        scope = '' if self.config.tf_scope is None else self.config.tf_scope + '-'
        self.hidden_layers = NeuralNetwork(self.config.network_layers, self.state,
                                           scope=scope + 'value_function')

        self.saver = tf.train.Saver()

        self.create_outputs()
        self.baseline_value_function = LinearValueFunction()
        self.create_training_operations()

        self.session.run(tf.global_variables_initializer())

    def create_outputs(self):
        # Output action means and log standard deviations

        with tf.variable_scope("policy"):
            self.action_means = linear(self.hidden_layers.get_output(),
                                      {'neurons': self.action_count}, 'action_mu')

            # Random init for log standard deviations
            log_standard_devs_init = tf.Variable(0.01 * self.random.randn(1, self.action_count),
                                                 dtype=tf.float32)

            self.action_log_stds = tf.tile(log_standard_devs_init, tf.pack((tf.shape(self.action_means)[0], 1)))

    def create_training_operations(self):
        """
        Creates TRPO training operations, i.e. the natural gradient update step
        based on the KL divergence constraint between new and old policy.
        :return:
        """

        with tf.variable_scope("update"):
            current_log_prob = get_log_prob_gaussian(self.action_means, self.action_log_stds, self.actions)
            prev_log_prob = get_log_prob_gaussian(self.prev_action_means, self.prev_action_log_stds, self.actions)
            prob_ratio = tf.exp(current_log_prob - prev_log_prob)

            surrogate_loss = -tf.reduce_mean(prob_ratio * self.advantage)
            variables = tf.trainable_variables()
            batch_float = tf.cast(self.batch_size, tf.float32)

            mean_kl_divergence = get_kl_divergence_gaussian(self.prev_action_means, self.prev_action_log_stds,
                                                            self.action_means, self.action_log_stds) / batch_float
            mean_entropy = get_entropy_gaussian(self.action_log_stds) / batch_float

            self.losses = [surrogate_loss, mean_kl_divergence, mean_entropy]

            # Get symbolic gradient expressions
            self.policy_gradient = get_flattened_gradient(self.losses, variables)
            fixed_kl_divergence = get_fixed_kl_divergence_gaussian(self.action_means, self.action_log_stds) \
                                  / batch_float

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

    def get_action(self, state, episode=1):
        """

        :param state: State tensor
        :return: Action and network output
        """

        action_means, action_log_stds = self.session.run([self.action_means,
                                                          self.action_log_stds],
                                                         {self.state: [state]})
        std = np.exp(action_log_stds) * self.random.randn(*action_log_stds.shape)
        print('action_means =' + str(action_means))
        print('std=' + str(std))
        action = action_means # + std
        return action.ravel(), dict(action_means=action_means,
                                    action_log_stds=action_log_stds)

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the constrained optimisation based on the fixed kl-divergence constraint.

        :param batch:
        :return:
        """
        # Set per episode advantage using GAE
        self.input_feed = None
        self.compute_gae_advantage(batch, self.gamma, self.gae_lambda, self.use_gae)

        # Update linear value function for baseline prediction
        self.baseline_value_function.fit(batch)

        # Merge episode inputs into single arrays
        action_log_stds, action_means, actions, batch_advantage, states = self.merge_episodes(batch)

        self.input_feed = {self.state: states,
                           self.actions: actions,
                           self.advantage: batch_advantage,
                           self.prev_action_means: action_means,
                           self.prev_action_log_stds: action_log_stds}

        previous_theta = self.flat_variable_helper.get()
        gradient = self.session.run(self.policy_gradient, self.input_feed)
        zero = np.zeros_like(gradient)
        if np.allclose(gradient, zero):
            print('Gradient zero, skipping update')
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
            if improved:
                print('Updating with line search result..')
                self.flat_variable_helper.set(theta)
            else:
                print('Updating with full step..')
                self.flat_variable_helper.set(previous_theta + update_step)

            # Compute full update based on line search result
            surrogate_loss, kl_divergence, entropy = self.session.run(self.losses, self.input_feed)

            print('Surrogate loss=' + str(surrogate_loss))
            print('KL-divergence after update=' + str(kl_divergence))
            print('Entropy=' + str(entropy))

    def compute_fvp(self, p):
        self.input_feed[self.flat_tangent] = p

        return self.session.run(self.fisher_vector_product, self.input_feed) + p * self.cg_damping

    def compute_surrogate_loss(self, theta):
        self.flat_variable_helper.set(theta)

        # Losses[0] = surrogate_loss
        return self.session.run(self.losses[0], self.input_feed)
