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

Based on https://github.com/ilyasu123/trpo, with a slightly more readable
modularisation.

"""
from tensorforce.config import create_config
from tensorforce.neural_networks.layers import dense
from tensorforce.neural_networks.neural_network import NeuralNetwork
from tensorforce.updater.value_function import ValueFunction
from tensorforce.util.experiment_util import global_seed
from tensorforce.util.math_util import *

import numpy as np
import tensorflow as tf


class TRPOUpdater(ValueFunction):
    default_config = {
        'cg_damping': 0.1,
        'max_kl_divergence': 0.01,
    }

    def __init__(self, config):
        super(TRPOUpdater, self).__init__(config)

        self.config = create_config(config, default=self.default_config)
        self.batch_size = config.batch_size
        self.action_count = self.config.actions
        self.cg_damping = self.config.cg_damping
        self.max_kl_divergence = self.config.max_kl_divergence

        self.gamma = self.config.gamma

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        if self.config.concat is not None and self.config.concat > 1:
            self.state = tf.placeholder(tf.float32, [None, self.config.concat_length] + list(self.config.state_shape),
                                        name="state")
        else:
            self.state = tf.placeholder(tf.float32, [None] + list(self.config.state_shape), name="state")

        self.episode = 0

        self.actions = tf.placeholder(tf.float32, [None, self.action_count], name='actions')
        self.advantage = tf.placeholder(tf.float32, shape=[None])
        self.flat_tangent = tf.placeholder(tf.float32, shape=[None])
        self.prev_action_means = tf.placeholder(tf.float32, [None, self.action_count])
        self.prev_action_log_stds = tf.placeholder(tf.float32, [None, self.action_count])

        self.hidden_layers = NeuralNetwork(self.config.network_layers, self.state, 'value_function')

        self.variables = tf.trainable_variables()
        self.flat_variable_helper = FlatVarHelper(self.session, self.variables)
        self.saver = tf.train.Saver()

        self.create_outputs()
        self.create_training_operations()
        self.linear_value_function = LinearValueFunction()

        self.session.run(tf.initialize_all_variables())

    def create_outputs(self):
        # Output action means and log standard deviations

        with tf.variable_scope("policy"):
            self.action_means = dense(self.hidden_layers.get_output(),
                                      {'neurons': self.action_count, 'regularization': self.config.regularizer,
                                       'regularization_param': self.config.regularization_param}, 'action_mu')

            # Random init for log standard deviations
            log_standard_devs_init = tf.Variable(0.01 * self.random.randn(1, self.action_count))

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
            mean_kl_divergence = get_kl_divergence_gaussian(self.prev_action_means, self.prev_action_log_stds,
                                                            self.action_means, self.action_log_stds) / float(
                self.batch_size)
            mean_entropy = get_entropy_gaussian(self.action_log_stds) / float(self.batch_size)

            self.losses = [surrogate_loss, mean_kl_divergence, mean_entropy]

            # Get symbolic gradient expressions
            self.policy_gradient = get_flattened_gradient(self.losses, self.variables)

            # Natural gradient update
            fixed_kl_divergence = get_fixed_kl_divergence_gaussian(self.action_means, self.action_log_stds) \
                                  / float(self.batch_size)

            variable_shapes = map(get_shape, self.variables)
            offset = 0
            tangents = []
            gradients = tf.gradients(fixed_kl_divergence, self.variables)

            for shape in variable_shapes:
                size = np.prod(shape)
                param = tf.reshape(self.flat_tangent[offset:(offset + size)], shape)
                tangents.append(param)

                offset += size

            gradient_vector_product = [tf.reduce_sum(g * t) for (g, t) in zip(gradients, tangents)]

            self.fisher_vector_product = get_flattened_gradient(gradient_vector_product, self.variables)

    def get_action(self, state):
        """

        :param state: State tensor
        :return: Action and network output
        """

        action_means, action_log_stds = self.session.run([self.action_means,
                                                          self.action_log_stds],
                                                         {self.state: state})

        action = action_means + np.exp(action_log_stds) * self.random.randn(*action_log_stds.shape)

        return action, dict(action_means=action_means,
                            action_log_stds=action_log_stds)

    def update(self, batch):
        """
        Compute update for one batch of experiences.
        :param batch:
        :return:
        """

        # Estimate per episode advantages and returns through linear baseline estimate
        for path in batch:
            path['baseline'] = self.linear_value_function.predict(path)
            path['returns'] = discount(path['rewards'], self.gamma)
            path['advantage'] = path['returns'] - path['baseline']

        action_means = np.concatenate([path['action_means'] for path in batch])
        action_log_stds = np.concatenate([path['action_log_stds'] for path in batch])

        states = np.concatenate([path['states'] for path in batch])
        actions = np.concatenate([path['actions'] for path in batch])
        advantages = [path['advantages'] for path in batch]
        advantages = zero_mean_unit_variance(advantages)

        self.linear_value_function.fit(batch)

        input_feed = {self.state: states,
                      self.actions: actions,
                      self.advantage: advantages,
                      self.prev_action_means: action_means,
                      self.prev_action_log_stds: action_log_stds}

        previous_theta = self.flat_variable_helper.get()

        def fisher_vector_product(p):
            input_feed[self.flat_tangent] = p

            return self.session.run(self.fisher_vector_product, input_feed) + p * self.cg_damping

        gradient = self.session.run(self.policy_gradient, feed_dict=input_feed)
        cg_direction = conjugate_gradient(fisher_vector_product, -gradient)

        shs = (.5 * cg_direction.dot(fisher_vector_product(cg_direction)))

        lagrange_multiplier = np.sqrt(shs / self.max_kl_divergence)

        update_step = cg_direction / lagrange_multiplier
        #negative_gradient_direction = -g.dot(cg_direction)

        def loss(theta):
            self.flat_variable_helper.set(theta)
            return self.session.run(self.losses[0], feed_dict=input_feed)

        # TODO do we need line search in practice?

        theta = previous_theta + update_step
        self.flat_variable_helper.set(theta)
        self.session.run(self.losses, feed_dict=input_feed)
