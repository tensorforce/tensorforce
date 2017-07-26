# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Implements proximal policy optimization with general advantage estimation (PPO-GAE) as
introduced by Schulman et al.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from six.moves import xrange
from tensorforce import util
from tensorforce.core.memories import Replay
from tensorforce.models import PolicyGradientModel


class PPOModel(PolicyGradientModel):

    allows_discrete_actions = True
    allows_continuous_actions = True

    default_config = dict(
        entropy_penalty=0.01,
        loss_clipping=0.1,  # Trust region clipping
        epochs=10,  # Number of training epochs for SGD,
        optimizer_batch_size=100,  # Batch size for optimiser
        random_sampling=True # Sampling strategy for replay memory
    )

    def __init__(self, config):
        config.default(PPOModel.default_config)
        super(PPOModel, self).__init__(config)
        self.epochs = config.epochs
        self.optimizer_batch_size = config.optimizer_batch_size
        # Use replay memory so memory logic can be used to sample batches
        self.memory = Replay(config.batch_size, config.states, config.actions, config.random_sampling)

    def create_tf_operations(self, config):
        """
        Creates PPO training operations, i.e. the SGD update
        based on the trust region loss.
        :return:
        """
        super(PPOModel, self).create_tf_operations(config)

        with tf.variable_scope('update'):
            prob_ratios = list()
            entropy_penalties = list()
            kl_divergences = list()

            for name, action in self.action.items():
                distribution = self.distribution[name]
                prev_distribution = tuple(tf.placeholder(dtype=tf.float32, shape=util.shape(x, unknown=None)) for x in distribution)
                self.internal_inputs.extend(prev_distribution)
                self.internal_outputs.extend(distribution)
                self.internal_inits.extend(np.zeros(shape=util.shape(x)[1:]) for x in distribution)
                prev_distribution = self.distribution[name].__class__.from_tensors(parameters=prev_distribution, deterministic=self.deterministic)

                # Standard policy gradient log likelihood computation
                log_prob = distribution.log_probability(action=action)
                prev_log_prob = prev_distribution.log_probability(action=action)
                log_prob_diff = tf.minimum(x=(log_prob - prev_log_prob), y=10.0)

                prob_ratio = tf.exp(x=log_prob_diff)

                entropy = distribution.entropy()
                entropy_penalty = -config.entropy_penalty * entropy

                kl_divergence = distribution.kl_divergence(prev_distribution)

                prs_list = [prob_ratio]
                eps_list = [entropy_penalty]
                kds_list = [kl_divergence]
                for _ in range(len(config.actions[name].shape)):
                    prs_list = [pr for prs in prs_list for pr in tf.unstack(value=prs, axis=1)]
                    eps_list = [ep for eps in eps_list for ep in tf.unstack(value=eps, axis=1)]
                    kds_list = [kd for kds in kds_list for kd in tf.unstack(value=kds, axis=1)]
                prob_ratios.extend(prs_list)
                entropy_penalties.extend(eps_list)
                kl_divergences.extend(kds_list)

            # The surrogate loss in PPO is the minimum of clipped loss and
            # target advantage * prob_ratio, which is the CPO loss
            # Presentation on conservative policy iteration:
            # https://www.cs.cmu.edu/~jcl/presentation/RL/RL.ps
            prob_ratio = tf.add_n(inputs=prob_ratios) / len(prob_ratios)
            prob_ratio = tf.clip_by_value(prob_ratio, 1.0 - config.loss_clipping, 1.0 + config.loss_clipping)

            self.loss_per_instance = -prob_ratio * self.reward
            self.surrogate_loss = tf.reduce_mean(input_tensor=self.loss_per_instance, axis=0)
            tf.losses.add_loss(self.surrogate_loss)

            self.entropy_penalty = tf.reduce_mean(input_tensor=(tf.add_n(inputs=entropy_penalties) / len(entropy_penalties)), axis=0)
            tf.losses.add_loss(self.entropy_penalty)
            # Note: Not computing the trust region loss on the value function because
            # the value function does not share a network with the policy. Worth
            # analysing how this impacts performance.

            self.kl_divergence = tf.reduce_mean(input_tensor=(tf.add_n(inputs=kl_divergences) / len(kl_divergences)), axis=0)

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the trust region update based on SGD on the clipped loss.

        :param batch: On policy batch of experiences.
        :return:
        """

        # Compute GAE.
        self.advantage_estimation(batch)

        if self.baseline:
            self.baseline.update(states=batch['states'], returns=batch['returns'])

        # Set memory contents to batch contents
        self.memory.set_memory(
            states=batch['states'],
            actions=batch['actions'],
            rewards=batch['rewards'],
            terminals=batch['terminals'],
            internals=batch['internals']
        )

        # PPO takes multiple passes over the on-policy batch.
        # We use a memory sampling random ranges (as opposed to keeping
        # track of indices and e.g. first taking elems 0-15, then 16-32, etc).
        for epoch in xrange(self.epochs):
            self.logger.debug('Optimising PPO, epoch = {}'.format(epoch))

            # Sample a batch by sampling a starting point and taking a range from there.
            batch = self.memory.get_batch(self.optimizer_batch_size)

            fetches = [self.optimize, self.loss, self.loss_per_instance]

            feed_dict = {state: batch['states'][name] for name, state in self.state.items()}
            feed_dict.update({action: batch['actions'][name] for name, action in self.action.items()})
            feed_dict[self.reward] = batch['rewards']
            feed_dict[self.terminal] = batch['terminals']
            feed_dict.update({internal: batch['internals'][n] for n, internal in enumerate(self.internal_inputs)})

            # self.surrogate_loss, self.entropy_penalty, self.kl_divergence
            loss, loss_per_instance = self.session.run(fetches=fetches, feed_dict=feed_dict)[1:3]

            #self.logger.debug('Loss = {}'.format(loss))
            #self.logger.debug('KL divergence = {}'.format(kl_divergence))
            #self.logger.debug('Entropy = {}'.format(entropy))

        # TODO: average instead of last iteration?
        return loss, loss_per_instance
