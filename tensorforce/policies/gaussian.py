from tensorforce.policies.policy import Policy
import tensorflow as tf
import numpy as np

class Gaussian(Policy):

    def entropy(self, dist):
        return tf.reduce_sum(log_std + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))

    def log_prob(self, dist, action):
        pass

    def kl_divergence(self, dist_a, dist_b):
        pass