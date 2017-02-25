# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

from tensorforce.models.policies.distribution import Distribution
import numpy as np
import tensorflow as tf


class Categorical(Distribution):
    def __init__(self, random):
        super(Categorical, self).__init__(random)

    def kl_divergence(self, dist_a, dist_b):
        prob_a = dist_a['policy_output']
        prob_b = dist_b['policy_output']

        # Need to ensure numerical stability
        return tf.reduce_sum(prob_a * tf.log((prob_a + self.epsilon) / (prob_b + self.epsilon)))

    def entropy(self, dist):
        prob = dist['policy_output']

        return -tf.reduce_sum(prob * tf.log((prob + self.epsilon)))

    def log_prob(self, dist, actions):
        prob = dist['policy_output']

        return tf.log(tf.reduce_sum(tf.multiply(prob, actions), [1]) + self.epsilon)

    def fixed_kl(self, dist):
        """
        KL divergence with first param fixed. Used in TRPO update.

        """
        prob = dist['policy_output']

        return tf.reduce_sum(tf.stop_gradient(prob)
                             * tf.log(tf.stop_gradient(prob + self.epsilon) / (prob + self.epsilon)))

    def sample(self, dist):
        prob = dist['policy_output']

        # Categorical dist is special case of multinomial
        return np.flatnonzero(self.random.multinomial(1, prob, 1))[0]
