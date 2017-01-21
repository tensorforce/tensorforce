"""
Standard Gaussian policy.
"""

from tensorforce.models.neural_networks.layers import linear
from tensorforce.models.policies.gaussian import Gaussian
from tensorforce.models.policies.stochastic_policy import StochasticPolicy
import numpy as np
import tensorflow as tf


class GaussianPolicy(StochasticPolicy):
    def __init__(self, neural_network=None,
                 session=None,
                 state=None,
                 random=None,
                 action_count=1,
                 scope='policy'):
        super(GaussianPolicy, self).__init__(neural_network, session, state, random, action_count)
        self.dist = Gaussian()

        with tf.variable_scope(scope):
            self.action_means = linear(self.neural_network.get_output(),
                                       {'neurons': self.action_count}, 'action_mu')

            # Random init for log standard deviations
            log_standard_devs_init = tf.Variable(0.01 * self.random.randn(1, self.action_count),
                                                 dtype=tf.float32)

            self.action_log_stds = tf.tile(log_standard_devs_init, tf.pack((tf.shape(self.action_means)[0], 1)))

    def sample(self, state):
        action_means, action_log_stds = self.session.run([self.action_means,
                                                          self.action_log_stds],
                                                         {self.state: [state]})

        action = action_means + np.exp(action_log_stds) * self.random.randn(*action_log_stds.shape)

        return action.ravel(), dict(action_means=action_means,
                                    action_log_stds=action_log_stds)

    def get_distribution(self):
        return self.dist

    def get_output_variables(self):
        return dict(policy_output=self.action_means,
                    policy_log_std=self.action_log_stds)
