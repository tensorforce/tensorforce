from tensorforce.models.policies.distribution import Distribution
import numpy as np
import tensorflow as tf


class Categorical(Distribution):

    def kl_divergence(self, dist_a, dist_b):
        return tf.reduce_sum(dist_a * tf.log(dist_a/dist_b))

    def entropy(self, dist):
        return -tf.reduce_sum(dist * tf.log(dist))

    def log_prob(self, dist, actions):
        tf.log(tf.reduce_sum(tf.mul(dist, actions), reduction_indices=[1]))

    def fixed_kl(self, dist_a, epsilon=1e-6):
        """
        KL divergence with first param fixed. Used in TRPO update.

        """
        return tf.reduce_sum(tf.stop_gradient(dist_a)
                             * tf.log(tf.stop_gradient(dist_a + epsilon) / (dist_a + epsilon)))

    def sample(self, output_dist):
        cumulative_prob = np.cumsum(output_dist, axis=1)

        return np.argmax(cumulative_prob > np.random.rand(output_dist.shape[0], 1), axis=1)