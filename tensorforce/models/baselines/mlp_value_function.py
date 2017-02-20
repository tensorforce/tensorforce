# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Multi-layer perceptron baseline value function
"""
from tensorforce.models.baselines.value_function import ValueFunction
import tensorflow as tf
import numpy as np

from tensorforce.models.neural_networks.layers import dense


class MLPValueFunction(ValueFunction):

    def __init__(self, session=None, update_iterations=100, layer_size=64):
        self.session = session
        self.mlp = None
        self.update_iterations = update_iterations
        self.layer_size = layer_size
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

    def predict(self, path):
        if self.mlp is None:
            return np.zeros(len(path["rewards"]))
        else:
            return self.session.run(self.mlp, {self.input: self.get_features(path)})

    def fit(self, paths):
        feature_matrix = np.concatenate([self.get_features(path) for path in paths])

        if self.mlp is None:
            self.create_net(feature_matrix.shape[1])

        returns = np.concatenate([path["returns"] for path in paths])

        for _ in range(self.update_iterations):
            self.session.run(self.update, {self.input: feature_matrix, self.labels: returns})

    def create_net(self, input_shape):
        with tf.variable_scope("mlp_value_function"):
            self.input = tf.placeholder(tf.float32, shape=[None, input_shape], name="input")

            hidden_1 = dense(self.input, {'num_outputs': input_shape}, 'hidden_1')
            hidden_2 = dense(hidden_1, {'num_outputs': self.layer_size}, 'hidden_2')
            out = dense(hidden_2, {'num_outputs': 1}, 'out')
            self.mlp = tf.reshape(out, (-1,))

            l2 = tf.nn.l2_loss(self.mlp - self.labels)
            self.update = tf.train.AdamOptimizer().minimize(l2)

            self.session.run(tf.initialize_all_variables())




