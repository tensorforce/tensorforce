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
https://github.com/ilyasu123/trpo

The core training update code is under MIT license, for more information see LICENSE-EXT.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorforce import util
from tensorforce.models import PolicyGradientModel
from tensorforce.core.optimizers import ConjugateGradientOptimizer


class TRPOModel(PolicyGradientModel):

    allows_discrete_actions = True
    allows_continuous_actions = True

    default_config = dict(
        optimizer=None,
        override_line_search=False,
        cg_damping=0.001,
        line_search_steps=20,
        max_kl_divergence=0.001,
        cg_iterations=20
    )

    def __init__(self, config):
        config.default(TRPOModel.default_config)
        super(TRPOModel, self).__init__(config)

        self.override_line_search = config.override_line_search
        self.cg_damping = config.cg_damping
        self.max_kl_divergence = config.max_kl_divergence
        self.line_search_steps = config.line_search_steps

    def create_tf_operations(self, config):
        """
        Creates TRPO training operations, i.e. the natural gradient update step
        based on the KL divergence constraint between new and old policy.
        :return:
        """
        super(TRPOModel, self).create_tf_operations(config)

        with tf.variable_scope('update'):
            prob_ratios = list()
            kl_divergences = list()
            entropies = list()
            fixed_kl_divergences = list()

            for name, action in self.action.items():
                distribution = self.distribution[name]
                prev_distribution = tuple(tf.placeholder(dtype=tf.float32, shape=util.shape(x, unknown=None)) for x in distribution)
                self.internal_inputs.extend(prev_distribution)
                self.internal_outputs.extend(distribution)
                self.internal_inits.extend(np.zeros(shape=util.shape(x)[1:]) for x in distribution)
                prev_distribution = distribution.from_tensors(parameters=prev_distribution, deterministic=self.deterministic)

                shape_size = util.prod(config.actions[name].shape)

                log_prob = distribution.log_probability(action=action)
                prev_log_prob = prev_distribution.log_probability(action=action)
                log_prob_diff = tf.minimum(x=(log_prob - prev_log_prob), y=10.0)
                prob_ratio = tf.exp(x=log_prob_diff)
                prob_ratio = tf.reshape(tensor=prob_ratio, shape=(-1, shape_size))
                prob_ratios.append(prob_ratio)

                kl_divergence = distribution.kl_divergence(other=prev_distribution)
                kl_divergence = tf.reshape(tensor=kl_divergence, shape=(-1, shape_size))
                kl_divergences.append(kl_divergence)

                entropy = distribution.entropy()
                entropy = tf.reshape(tensor=entropy, shape=(-1, shape_size))
                entropies.append(entropy)

                fixed_distribution = distribution.__class__.from_tensors(parameters=[tf.stop_gradient(x) for x in distribution], deterministic=self.deterministic)
                fixed_kl_divergence = fixed_distribution.kl_divergence(distribution)
                fixed_kl_divergence = tf.reshape(tensor=fixed_kl_divergence, shape=(-1, shape_size))
                fixed_kl_divergences.append(fixed_kl_divergence)

            prob_ratio = tf.reduce_mean(input_tensor=tf.concat(values=prob_ratios, axis=1), axis=1)
            self.loss_per_instance = -prob_ratio * self.reward
            surrogate_loss = tf.reduce_mean(input_tensor=self.loss_per_instance, axis=0)

            kl_divergence = tf.reduce_mean(input_tensor=tf.concat(values=kl_divergences, axis=1), axis=1)
            kl_divergence = tf.reduce_mean(input_tensor=kl_divergence, axis=0)
            entropy = tf.reduce_mean(input_tensor=tf.concat(values=entropies, axis=1), axis=1)
            entropy = tf.reduce_mean(input_tensor=entropy, axis=0)
            self.losses = (surrogate_loss, kl_divergence, entropy, self.loss_per_instance)

            fixed_kl_divergence = tf.reduce_mean(input_tensor=tf.concat(values=fixed_kl_divergences, axis=1), axis=1)

            # Get symbolic gradient expressions
            variables = list(tf.trainable_variables())  # TODO: ideally not value function (see also for "gradients" below)
            gradients = tf.gradients(self.losses[0], variables)
            variables = [var for var, grad in zip(variables, gradients) if grad is not None]
            gradients = [grad for grad in gradients if grad is not None]
            self.policy_gradient = tf.concat(values=[tf.reshape(grad, (-1,)) for grad in gradients], axis=0)  # util.prod(util.shape(v))

            self.tangent = tf.placeholder(tf.float32, shape=(None,))
            offset = 0
            tangents = []
            for variable in variables:
                shape = util.shape(variable)
                size = util.prod(shape)
                tangents.append(tf.reshape(self.tangent[offset:offset + size], shape))
                offset += size

            gradients = tf.gradients(fixed_kl_divergence, variables)
            gradient_vector_product = [tf.reduce_sum(g * t) for (g, t) in zip(gradients, tangents)]

            self.flat_variable_helper = FlatVarHelper(variables)
            gradients = tf.gradients(gradient_vector_product, variables)
            self.fisher_vector_product = tf.concat(values=[tf.reshape(grad, (-1,)) for grad in gradients], axis=0)

            self.cg_optimizer = ConjugateGradientOptimizer(self.logger, config.cg_iterations)

    def set_session(self, session):
        super(TRPOModel, self).set_session(session)
        self.flat_variable_helper.session = session

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the constrained optimisation based on the fixed kl-divergence constraint.

        :param batch:
        :return:
        """
        super(TRPOModel, self).update(batch)

        self.feed_dict = {state: batch['states'][name] for name, state in self.state.items()}
        self.feed_dict.update({action: batch['actions'][name] for name, action in self.action.items()})
        self.feed_dict[self.reward] = batch['rewards']
        self.feed_dict[self.terminal] = batch['terminals']
        self.feed_dict.update({internal: batch['internals'][n] for n, internal in enumerate(self.internal_inputs)})

        gradient = self.session.run(self.policy_gradient, self.feed_dict)  # dL

        if np.allclose(gradient, np.zeros_like(gradient)):
            self.logger.debug('Gradient zero, skipping update.')
            return

        # The details of the approximations used here to solve the constrained
        # optimisation can be found in Appendix C of the TRPO paper
        # Note that no subsampling is used, which would improve computational performance
        search_direction = self.cg_optimizer.solve(self.compute_fvp, -gradient)  # x = ddKL(=F)^(-1) * -dL

        # Search direction has now been approximated as cg-solution s= A^-1g where A is
        # Fisher matrix, which is a local approximation of the
        # KL divergence constraint
        shs = 0.5 * search_direction.dot(self.compute_fvp(search_direction))  # (c lambda^2) = 0.5 * xT * F * x
        if shs < 0:
            self.logger.debug('Computing search direction failed, skipping update.')
            return

        lagrange_multiplier = np.sqrt(shs / self.max_kl_divergence)
        update_step = search_direction / (lagrange_multiplier + util.epsilon)  # c
        negative_gradient_direction = -gradient.dot(search_direction)  # -dL * x

        # Improve update step through simple backtracking line search
        # N.b. some implementations skip the line search
        previous_theta = self.flat_variable_helper.get()
        improved, theta = line_search(self.compute_surrogate_loss, previous_theta, update_step, negative_gradient_direction / (lagrange_multiplier + util.epsilon), self.line_search_steps)

        # Use line search results, otherwise take full step
        # N.B. some implementations don't use the line search
        if improved:
            self.logger.debug('Updating with line search result..')
            self.flat_variable_helper.set(theta)
        elif self.override_line_search:
            self.logger.debug('Updating with full step..')
            self.flat_variable_helper.set(previous_theta + update_step)
        else:
            self.logger.debug('Failed to find line search solution, skipping update.')

        # Get loss values for progress monitoring
        surrogate_loss, kl_divergence, entropy, loss_per_instance = self.session.run(self.losses, self.feed_dict)

        # Sanity checks. Is entropy decreasing? Is KL divergence within reason? Is loss non-zero?
        self.logger.debug('Surrogate loss = {}'.format(surrogate_loss))
        self.logger.debug('KL-divergence after update = {}' .format(kl_divergence))
        self.logger.debug('Entropy = {}'.format(entropy))

        return (surrogate_loss, kl_divergence, entropy), loss_per_instance

    def compute_fvp(self, p):
        self.feed_dict[self.tangent] = p

        return self.session.run(self.fisher_vector_product, self.feed_dict) + p * self.cg_damping

    def compute_surrogate_loss(self, theta):
        self.flat_variable_helper.set(theta)

        # Losses[0] = surrogate_loss
        return self.session.run(self.losses[0], self.feed_dict)


class FlatVarHelper(object):

    def __init__(self, variables):
        self.session = None
        shapes = [util.shape(variable) for variable in variables]
        total_size = sum(util.prod(shape) for shape in shapes)
        self.theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []

        for (shape, variable) in zip(shapes, variables):
            size = util.prod(shape)
            assigns.append(tf.assign(variable, tf.reshape(self.theta[start:start + size], shape)))
            start += size

        self.set_op = tf.group(*assigns)
        self.get_op = tf.concat(axis=0, values=[tf.reshape(variable, (-1,)) for variable in variables])

    def set(self, theta):
        """
        Assign flat variable representation.

        :param theta: values
        """
        self.session.run(self.set_op, feed_dict={self.theta: theta})

    def get(self):
        """
        Get flat representation.

        :return: Concatenation of variables
        """

        return self.session.run(self.get_op)


def line_search(f, initial_x, full_step, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    """
    Line search for TRPO where a full step is taken first and then backtracked to
    find optimal step size.

    :param f:
    :param initial_x:
    :param full_step:
    :param expected_improve_rate:
    :param max_backtracks:
    :param accept_ratio:
    :return:
    """

    function_value = f(initial_x)

    for _, step_fraction in enumerate(0.5 ** np.arange(max_backtracks)):
        updated_x = initial_x + step_fraction * full_step
        new_function_value = f(updated_x)

        actual_improve = function_value - new_function_value
        expected_improve = expected_improve_rate * step_fraction

        improve_ratio = actual_improve / (expected_improve + util.epsilon)

        if improve_ratio > accept_ratio and actual_improve > 0:
            return True, updated_x

    return False, initial_x
