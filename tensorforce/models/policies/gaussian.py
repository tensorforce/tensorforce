import numpy as np
import tensorflow as tf

from tensorforce.models.policies.distribution import Distribution


class Gaussian(Distribution):

    def log_prob(self, dist, actions=0):
        mean = dist['policy_output']
        log_std = dist['policy_log_std']

        probability = -tf.square(actions - mean) / (2 * tf.exp(2 * log_std)) \
                      - 0.5 * tf.log(tf.constant(2 * np.pi)) - log_std

        # Sum logs
        return tf.reduce_sum(probability, [1])

    def kl_divergence(self, dist_a, dist_b,):
        mean_a = dist_a['policy_output']
        log_std_a = dist_a['policy_log_std']

        mean_b= dist_b['policy_output']
        log_std_b = dist_b['policy_log_std']

        exp_std_a = tf.exp(2 * log_std_a)
        exp_std_b = tf.exp(2 * log_std_b)

        return tf.reduce_sum(log_std_b - log_std_a
                             + (exp_std_a + tf.square(mean_a - mean_b)) / (2 * exp_std_b) - 0.5)

    def entropy(self, dist):
        log_std = dist['policy_log_std']

        return tf.reduce_sum(log_std + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))

    def fixed_kl(self, dist):
        """
        KL divergence with first param fixed.

        :param mean:
        :param log_std:
        :return:
        """
        mean = dist['policy_output']
        log_std = dist['policy_log_std']

        mean_a, log_std_a = map(tf.stop_gradient, [mean, log_std])

        dist_a = dict(policy_output=mean_a,
                      policy_log_std=log_std_a)

        return self.kl_divergence(dist_a, dist)