from tensorforce.models.policies.distribution import Distribution
import numpy as np
import tensorflow as tf


class Categorical(Distribution):

    def __init__(self):
        super(Categorical, self).__init__()

    def kl_divergence(self, dist_a, dist_b):
        prob_a = dist_a['policy_output']
        prob_b = dist_b['policy_output']

        # Need to ensure numerical stability
        return tf.reduce_sum(prob_a * tf.log((prob_a + self.epsilon) /(prob_b + self.epsilon)))

    def entropy(self, dist):
        prob = dist['policy_output']

        return -tf.reduce_sum(prob * tf.log((prob + self.epsilon)))

    def log_prob(self, dist, actions):
        prob = dist['policy_output']

        return tf.log(tf.reduce_sum(tf.mul((prob + self.epsilon), actions), [1]))


    def fixed_kl(self, dist):
        """
        KL divergence with first param fixed. Used in TRPO update.

        """
        prob = dist['policy_output']

        return tf.reduce_sum(tf.stop_gradient(prob)
                             * tf.log(tf.stop_gradient(prob + self.epsilon) / (prob + self.epsilon)))

    def sample(self, dist):
        prob = dist['policy_output']

        cumulative_prob = np.cumsum(prob, axis=0)
        print(cumulative_prob)
        samples = cumulative_prob > np.random.rand(len(dist), 1)

        return np.argmax(samples.ravel())