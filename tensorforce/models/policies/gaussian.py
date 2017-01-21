import numpy as np
import tensorflow as tf

from tensorforce.models.policies import Policy
from tensorforce.models.policies.distribution import Distribution


class Gaussian(Distribution):

    def log_prob(self, mean=0, log_std=0, actions=0):
        probability = -tf.square(actions - mean) / (2 * tf.exp(2 * log_std)) \
                      - 0.5 * tf.log(tf.constant(2 * np.pi)) - log_std

        # Sum logs
        return tf.reduce_sum(probability, [1])

    def kl_divergence(self, mean_a=0, log_std_a=0, mean_b=0, log_std_b=0,):
        exp_std_a = tf.exp(2 * log_std_a)
        exp_std_b = tf.exp(2 * log_std_b)

        return tf.reduce_sum(log_std_b - log_std_a
                             + (exp_std_a + tf.square(mean_a - mean_b)) / (2 * exp_std_b) - 0.5)

    def entropy(self, log_std=0):
        return tf.reduce_sum(log_std + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))

    def fixed_kl(self, mean, log_std):
        """
        KL divergence with first param fixed.

        :param mean:
        :param log_std:
        :return:
        """
        mean_a, log_std_a = map(tf.stop_gradient, [mean, log_std])
        mean_b, log_std_b = mean, log_std

        return self.kl_divergence(mean_a, log_std_a, mean_b, log_std_b)