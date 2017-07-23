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
        loss_clipping=0.1, # Trust region clipping
        epochs=10,  # Number of training epochs for SGD,
        optimizer_batch_size=100  # Batch size for optimiser
    )

    def __init__(self, config):
        config.default(PPOModel.default_config)
        super(PPOModel, self).__init__(config)
        self.epochs = config.epochs
        self.optimizer_batch_size = config.optimizer_batch_size
        self.config = config

    def create_tf_operations(self, config):
        """
        Creates PPO training operations, i.e. the SGD update
        based on the trust region loss.
        :return:
        """
        super(PPOModel, self).create_tf_operations(config)

        with tf.variable_scope('update'):
            losses = list()

            self.ppo_opt = []
            for name, action in config.actions:
                distribution = self.distribution[name]
                prev_distribution = tuple(tf.placeholder(dtype=tf.float32, shape=util.shape(x, unknown=None)) for x in distribution)
                self.internal_inputs.extend(prev_distribution)
                self.internal_outputs.extend(distribution)

                if len(distribution.distribution) == 2:
                    for n, x in enumerate(distribution):
                        if n == 0:
                            self.internal_inits.append(np.zeros(shape=util.shape(x)[1:]))
                        else:
                            self.internal_inits.append(np.ones(shape=util.shape(x)[1:]))
                else:
                    self.internal_inits.extend(np.zeros(shape=util.shape(x)[1:]) for x in distribution)
                distr_cls = self.distribution[name].__class__
                prev_distribution = distr_cls.from_tensors(parameters=prev_distribution)

                # Standard policy gradient log likelihood computation
                log_prob = distribution.log_probability(action=self.action[name])
                previous_log_prob = prev_distribution.log_probability(action=self.action[name])
                prob_ratio = tf.minimum(tf.exp(log_prob - previous_log_prob), 1000)

                entropy = distribution.entropy()
                entropy_penalty = -config.entropy_penalty * entropy

                # The surrogate loss in PPO is the minimum of clipped loss and
                # target advantage * prob_ratio, which is the CPO loss
                # Presentation on conservative policy iteration:
                # https://www.cs.cmu.edu/~jcl/presentation/RL/RL.ps
                self.loss_per_instance = tf.multiply(x=prob_ratio, y=self.reward)

                tf.losses.add_loss(self.loss_per_instance)
                clipped_loss = tf.clip_by_value(prob_ratio, 1.0 - config.loss_clipping,
                                                1.0 + config.loss_clipping) * self.reward

                surrogate_loss = -tf.reduce_mean(tf.minimum(self.loss_per_instance,
                                                            clipped_loss), axis=0)
                penalized_loss = surrogate_loss + entropy_penalty
                kl_divergence = distribution.kl_divergence(prev_distribution)
                losses.append((surrogate_loss, entropy_penalty, kl_divergence, entropy))

                # Note: Not computing the trust region loss on the value function because
                # the value function does not share a network with the policy. Worth
                # analysing how this impacts performance.

                # Performing SGD on this loss
                grads_and_vars = self.optimizer.compute_gradients(penalized_loss)
                self.ppo_opt.append(self.optimizer.apply_gradients(grads_and_vars))

        # Compute means over surrogate loss, entropy penalty kl_divergence, entropy
        # over actions
        self.losses = [tf.reduce_mean(loss) for loss in zip(*losses)]

    def set_session(self, session):
        super(PPOModel, self).set_session(session)

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the trust region update based on SGD on the clipped loss.

        :param batch: On policy batch of experienecs.
        :return:
        """

        # Compute GAE
        batch['returns'] = util.cumulative_discount(rewards=batch['rewards'], terminals=batch['terminals'],
                                                    discount=self.discount)
        batch['rewards'] = self.advantage_estimation(batch)

        if self.baseline:
            self.baseline.update(states=batch['states'], returns=batch['returns'])

        # Create a replay memory, then set memory contents to batch
        # contents so memory logic can be used to sample batches
        memory = Replay(self.config.batch_size, self.config.states, self.config.actions)
        memory.set_memory(batch['states'], batch['actions'], batch['rewards'], batch['terminals'],
                          batch['internals'])

        # PPO takes multiple passes over the on-policy batch.
        # We use a memory sampling random ranges (as opposed to keeping
        # track of indices and e.g. first taking elems 0-15, then 16-32, etc).
        for i in xrange(self.epochs):
            self.logger.debug('Optimising PPO, epoch = {}'.format(i))

            # Sample a batch
            minibatch = memory.get_batch(self.optimizer_batch_size)

            # Create inputs over named states and actions
            self.feed_dict = {state: minibatch['states'][name] for name, state in self.state.items()}
            self.feed_dict.update({action: minibatch['actions'][name] for name, action in self.action.items()})
            self.feed_dict[self.reward] = minibatch['rewards']
            self.feed_dict[self.terminal] = minibatch['terminals']
            self.feed_dict.update(
                {internal: minibatch['internals'][n] for n, internal in enumerate(self.internal_inputs)})

            # TODO: loss per instance is many items, so result list is too large
            fetched = self.session.run(self.ppo_opt + self.losses + [self.loss_per_instance], self.feed_dict)

            #print(type(fetched))
            #print(len(fetched))
            #print(fetched)
            #self.logger.debug('Loss = {}'.format(loss))
            #self.logger.debug('KL divergence = {}'.format(kl_divergence))
            #self.logger.debug('Entropy = {}'.format(entropy))