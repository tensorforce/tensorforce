"""
Deep Q network. Implements training and update logic as described
in the DQN paper.
"""
import tensorflow as tf


class DeepQNetwork(object):

    def __init__(self, config):
        self.training_network = None

    def evaluate(self, state):
        pass

    def update(self, batch_size):
        pass
