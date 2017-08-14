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
        optimizer_batch_size=128,  # Batch size for optimiser
        random_sampling=True  # Sampling strategy for replay memory
    )

    def __init__(self, config):
        config.default(PPOModel.default_config)
        super(PPOModel, self).__init__(config)
        self.optimizer_batch_size = config.optimizer_batch_size
        # Use replay memory so memory logic can be used to sample batches

        self.updates = int(config.batch_size / self.optimizer_batch_size) * config.epochs
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
            entropies = list()

            for name, action in self.action.items():
                shape_size = util.prod(config.actions[name].shape)
                distribution = self.distribution[name]
                fixed_distribution = distribution.__class__.from_tensors(
                    tensors=[tf.stop_gradient(x) for x in distribution.get_tensors()],
                    deterministic=self.deterministic
                )

                # Standard policy gradient log likelihood computation
                log_prob = distribution.log_probability(action=action)
                fixed_log_prob = fixed_distribution.log_probability(action=action)
                log_prob_diff = log_prob - fixed_log_prob
                prob_ratio = tf.exp(x=log_prob_diff)
                prob_ratio = tf.reshape(tensor=prob_ratio, shape=(-1, shape_size))
                prob_ratios.append(prob_ratio)

                entropy = distribution.entropy()
                entropy_penalty = -config.entropy_penalty * entropy
                entropy_penalty = tf.reshape(tensor=entropy_penalty, shape=(-1, shape_size))
                entropy_penalties.append(entropy_penalty)

                entropy = tf.reshape(tensor=entropy, shape=(-1, shape_size))
                entropies.append(entropy)

                kl_divergence = fixed_distribution.kl_divergence(other=distribution)
                kl_divergence = tf.reshape(tensor=kl_divergence, shape=(-1, shape_size))
                kl_divergences.append(kl_divergence)

            # The surrogate loss in PPO is the minimum of clipped loss and
            # target advantage * prob_ratio, which is the CPO loss
            # Presentation on conservative policy iteration:
            # https://www.cs.cmu.edu/~jcl/presentation/RL/RL.ps
            prob_ratio = tf.reduce_mean(input_tensor=tf.concat(values=prob_ratios, axis=1), axis=1)
            prob_ratio = tf.clip_by_value(prob_ratio, 1.0 - config.loss_clipping, 1.0 + config.loss_clipping)
            self.loss_per_instance = -prob_ratio * self.reward
            self.surrogate_loss = tf.reduce_mean(input_tensor=self.loss_per_instance, axis=0)
            tf.losses.add_loss(self.surrogate_loss)

            # Mean over actions, mean over batch
            entropy_penalty = tf.reduce_mean(input_tensor=tf.concat(values=entropy_penalties, axis=1), axis=1)
            self.entropy_penalty = tf.reduce_mean(input_tensor=entropy_penalty, axis=0)
            tf.losses.add_loss(self.entropy_penalty)

            entropy = tf.reduce_mean(input_tensor=tf.concat(values=entropies, axis=1), axis=1)
            self.entropy = tf.reduce_mean(input_tensor=entropy, axis=0)

            kl_divergence = tf.reduce_mean(input_tensor=tf.concat(values=kl_divergences, axis=1), axis=1)
            self.kl_divergence = tf.reduce_mean(input_tensor=kl_divergence, axis=0)

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the trust region update based on SGD on the clipped loss.

        :param batch: On policy batch of experiences.
        :return:
        """

        batch['rewards'], discounted_rewards = self.reward_estimation(
            states=batch['states'],
            rewards=batch['rewards'],
            terminals=batch['terminals']
        )
        if self.baseline:
            self.baseline.update(
                states=batch['states'],
                returns=discounted_rewards
            )

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
        for i in xrange(self.updates):
            self.logger.debug('Optimising PPO, update = {}'.format(i))
            batch = self.memory.get_batch(self.optimizer_batch_size)

            fetches = [self.optimize, self.loss, self.loss_per_instance, self.kl_divergence, self.entropy]

            feed_dict = {state: batch['states'][name] for name, state in self.state.items()}
            feed_dict.update({action: batch['actions'][name] for name, action in self.action.items()})
            feed_dict[self.reward] = batch['rewards']
            feed_dict[self.terminal] = batch['terminals']
            feed_dict.update({internal: batch['internals'][n] for n, internal in enumerate(self.internal_inputs)})

            # self.surrogate_loss, self.entropy_penalty, self.kl_divergence
            loss, loss_per_instance, kl_divergence, entropy = self.session.run(fetches=fetches, feed_dict=feed_dict)[1:5]

            self.logger.debug('Loss = {}'.format(loss))
            self.logger.debug('KL divergence = {}'.format(kl_divergence))
            self.logger.debug('Entropy = {}'.format(entropy))

        return loss, loss_per_instance
