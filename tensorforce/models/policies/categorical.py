from tensorforce.models.policies.distribution import Distribution
import numpy as np
import tensorflow as tf


class Categorical(Distribution):

    def kl_divergence(self, dist_a, dist_b):
        prob_a = dist_a['policy_output']
        prob_b = dist_b['policy_output']

        return tf.reduce_sum(prob_a * tf.log(prob_a/prob_b))

    def entropy(self, dist):
        prob = dist['policy_output']

        return -tf.reduce_sum(prob * tf.log(prob))

    def log_prob(self, dist, actions):
        prob = dist['policy_output']

        mul = tf.mul(prob, actions)
        return tf.log(tf.reduce_sum(mul, reduction_indices=[1]))

    def fixed_kl(self, dist, epsilon=1e-6):
        """
        KL divergence with first param fixed. Used in TRPO update.

        """
        prob = dist['policy_output']

        return tf.reduce_sum(tf.stop_gradient(prob)
                             * tf.log(tf.stop_gradient(prob + epsilon) / (prob + epsilon)))

    def sample(self, dist):
        prob = dist['policy_output']
        cumulative_prob = np.cumsum(prob, axis=0)

        return np.argmax(cumulative_prob > np.random.rand(prob.shape[0], 1), axis=0)