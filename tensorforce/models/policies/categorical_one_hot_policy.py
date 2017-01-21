"""
Categorial one hot policy, used for discrete policy gradients.
"""
from tensorforce.models.neural_networks.layers import linear
from tensorforce.models.policies.categorical import Categorical
from tensorforce.models.policies.stochastic_policy import StochasticPolicy
import numpy as np
import tensorflow as tf


class CategoricalOneHotPolicy(StochasticPolicy):
    def __init__(self, neural_network=None,
                 session=None,
                 state=None,
                 random=None,
                 action_count=1,
                 scope='policy'):
        super(CategoricalOneHotPolicy, self).__init__(neural_network, session, state, random, action_count)
        self.dist = Categorical()

        with tf.variable_scope(scope):
            self.outputs = linear(self.neural_network.get_output(),
                                  {'neurons': self.action_count, 'activation': tf.nn.softmax}, 'outputs')

    def get_distribution(self):
        return self.dist

    def sample(self, state, sample=True):
        output_dist = self.session.run(self.outputs, {self.state: [state]})

        if sample:
            action = int(self.dist.sample(output_dist))
        else:
            action = int(np.argmax(output_dist))

        return action, dict(policy_output=output_dist)

    def get_output_variables(self):
        return dict(policy_output=self.outputs)
