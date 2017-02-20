# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================
"""
Vanilla policy gradient implementation.
"""
import numpy as np
import tensorflow as tf

from tensorforce.models import LinearValueFunction
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.models.neural_networks.layers import linear
from tensorforce.models.pg_model import PGModel
from tensorforce.util.experiment_util import global_seed

from tensorforce.default_configs import VPGModelConfig

class VPGModel(PGModel):
    default_config = VPGModelConfig

    def __init__(self, config, scope):
        super(VPGModel, self).__init__(config, scope)

        self.create_training_operations()
        self.session.run(tf.global_variables_initializer())

    def create_training_operations(self):
        with tf.variable_scope("update"):
            self.log_probabilities = self.dist.log_prob(self.policy.get_policy_variables(), self.actions)

            # Concise: Get log likelihood of actions, weigh by advantages, compute gradient on that
            self.loss = -tf.reduce_mean(self.log_probabilities * self.advantage, name="loss_op")

            self.optimize_op = self.optimizer.minimize(self.loss)


    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the vanilla policy gradient.
        :param batch:
        :return:
        """
        # Set per episode advantage using GAE
        self.compute_gae_advantage(batch, self.gamma, self.gae_lambda)

        # Update linear value function for baseline prediction
        self.baseline_value_function.fit(batch)

        # Merge episode inputs into single arrays
        _, _, actions, batch_advantage, states = self.merge_episodes(batch)
        
        log_probs, loss, _ = self.session.run([self.log_probabilities, self.loss, self.optimize_op],
                                                     {self.state: states,
                                                      self.actions: actions,
                                                      self.advantage: batch_advantage})
       # print('log probs:' + str(log_probs))
       # print('loss:' + str(loss))
