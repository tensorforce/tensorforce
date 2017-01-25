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
        self.dist = Categorical(random)

        with tf.variable_scope(scope):
            self.action_layer = linear(self.neural_network.get_output(),
                                  {'neurons': self.action_count}, 'outputs')
            self.outputs = tf.nn.softmax(self.action_layer)

    def get_distribution(self):
        return self.dist

    def sample(self, state, sample=True):
        output_dist = self.session.run(self.outputs, {self.state: [state]})

#       print(output_dist)

        output_dist = output_dist.ravel()

        if sample:
            action = self.dist.sample(dict(policy_output=output_dist))
        else:
            action = int(np.argmax(output_dist))

 #       print('action after dist sample ' + str(action))

        one_hot = np.zeros_like(output_dist)
        one_hot[action] = 1

        # We return a one hot vector and then extract the concrete action in the pg agent
        return one_hot, dict(policy_output=output_dist)

    def get_policy_variables(self):
        return dict(policy_output=self.outputs)
