# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Implements trust region policy optimization with general advantage estimation (TRPO-GAE) as
introduced by Schulman et al.

Based on https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py and
https://github.com/ilyasu123/trpo, with a hopefully slightly more readable
modularisation and some modifications.

"""
import numpy as np
import tensorflow as tf

from tensorforce.models import LinearValueFunction
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.models.neural_networks.layers import linear
from tensorforce.models.pg_model import PGModel
from tensorforce.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from tensorforce.util.experiment_util import global_seed
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
        self.cg_optimizer = ConjugateGradientOptimizer(self.config.cg_iterations)

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
                           self.prev_action_means: action_means}
        if self.continuous:
            self.input_feed[self.prev_action_log_stds] = action_log_stds

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
            # N.B. some implementations don't use the line search
            if improved:
                print('Updating with line search result..')
                self.flat_variable_helper.set(theta)
            else:
                print('Updating with full step..')
                self.flat_variable_helper.set(previous_theta + update_step)

            # Get loss values for progress monitoring
            surrogate_loss, kl_divergence, entropy = self.session.run(self.losses, self.input_feed)

            # Sanity checks. Is entropy decreasing? Is KL divergence within reason? Is loss non-zero?
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
