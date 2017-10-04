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
import numpy as np

from tensorforce import util
from tensorforce.core.memories import Replay, PrioritizedReplay, Memory
from tensorforce.models import PolicyGradientModel


class PPOModel(PolicyGradientModel):
    allows_discrete_actions = True
    allows_continuous_actions = True

    default_config = dict(
        entropy_penalty=0.01,
        loss_clipping=0.2,  # Trust region clipping
        epochs=10,  # Number of training epochs for optimizer,
        optimizer_batch_size=64,  # Batch size for optimizer
        random_sampling=True  # Sampling strategy for replay memory
    )

    def __init__(self, config):
        config.default(PPOModel.default_config)
        super(PPOModel, self).__init__(config)
        self.optimizer_batch_size = config.optimizer_batch_size
        self.batch_size = config.batch_size
        self.updates = int(config.batch_size / self.optimizer_batch_size) * config.epochs

        # Use replay memory as a cache so it can be used to sample minibatches
        self.memory = Replay(config.batch_size, config.states, config.actions, config.random_sampling)

    def create_tf_operations(self, config):
        """
        Creates PPO training operations, i.e. the optimizer update
        based on the trust region loss.
        :return:
        """
        super(PPOModel, self).create_tf_operations(config)

        with tf.variable_scope('update'):
            prob_ratios = list()
            entropy_penalties = list()

            # for diagnostics
            kl_divergences = list()
            entropies = list()
            self.distribution_tensors = dict()
            self.prev_distribution_tensors = dict()

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

                self.distribution_tensors[name] = list(distribution.get_tensors())
                prev_distribution = list(tf.placeholder(dtype=tf.float32, shape=util.shape(tensor, unknown=None)) for tensor in distribution.get_tensors())
                self.prev_distribution_tensors[name] = prev_distribution
                prev_distribution = distribution.from_tensors(tensors=prev_distribution, deterministic=self.deterministic)

                kl_divergence = prev_distribution.kl_divergence(other=distribution)
                kl_divergence = tf.reshape(tensor=kl_divergence, shape=(-1, shape_size))
                kl_divergences.append(kl_divergence)

                entropy = tf.reshape(tensor=entropy, shape=(-1, shape_size))
                entropies.append(entropy)

            # The surrogate loss in PPO is the minimum of clipped loss and
            # target advantage * prob_ratio, which is the CPO loss
            # Presentation on conservative policy iteration:
            # https://www.cs.cmu.edu/~jcl/presentation/RL/RL.ps
            prob_ratio = tf.reduce_mean(input_tensor=tf.concat(values=prob_ratios, axis=1), axis=1)
            clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1.0 - config.loss_clipping, 1.0 + config.loss_clipping)
            self.loss_per_instance = -tf.minimum(x=(prob_ratio * self.reward), y=(clipped_prob_ratio * self.reward))
            self.surrogate_loss = tf.reduce_mean(input_tensor=self.loss_per_instance, axis=0)
            tf.losses.add_loss(self.surrogate_loss)

            # Mean over actions, mean over batch
            entropy_penalty = tf.reduce_mean(input_tensor=tf.concat(values=entropy_penalties, axis=1), axis=1)
            self.entropy_penalty = tf.reduce_mean(input_tensor=entropy_penalty, axis=0)
            tf.losses.add_loss(self.entropy_penalty)

            kl_divergence = tf.reduce_mean(input_tensor=tf.concat(values=kl_divergences, axis=1), axis=1)
            self.kl_divergence = tf.reduce_mean(input_tensor=kl_divergence, axis=0)

            entropy = tf.reduce_mean(input_tensor=tf.concat(values=entropies, axis=1), axis=1)
            self.entropy = tf.reduce_mean(input_tensor=entropy, axis=0)

    def update(self, batch):
        """
        Compute update for one batch of experiences using general advantage estimation
        and the trust region update based on optimizer on the clipped loss.

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
            minibatch = self.memory.get_batch(self.optimizer_batch_size)

            feed_dict = {state: minibatch['states'][name] for name, state in self.state.items()}
            feed_dict.update({action: minibatch['actions'][name] for name, action in self.action.items()})
            feed_dict[self.reward] = minibatch['rewards']
            feed_dict[self.terminal] = minibatch['terminals']
            feed_dict.update({internal: minibatch['internals'][n] for n, internal in enumerate(self.internal_inputs)})

            if i == 0:  # First update, fetch previous distribution tensors
                assert self.updates >= 2
                assert 'optimize' not in self.distribution_tensors
                fetches = dict(optimize=self.optimize)
                fetches.update(self.distribution_tensors)
                prev_distribution_tensors = self.session.run(fetches=fetches, feed_dict=feed_dict)
                prev_distribution_tensors.pop('optimize')
            else:  # Otherwise just optimize
                self.session.run(fetches=self.optimize, feed_dict=feed_dict)

        # For the last epoch, fetch return and diagnostics values for each instance
        # by sampling on seqential non-random ranges e.g 0-15, 16-32 etc
        losses = []
        losses_per_instance = []
        kl_divergences = []
        prev_distribution_tensors = {placeholder: tensor for name, placeholders in self.prev_distribution_tensors.items() for placeholder, tensor in zip(placeholders, prev_distribution_tensors[name])}
        for i in range(int(self.batch_size / self.optimizer_batch_size)):
            start, end = i * self.optimizer_batch_size, (i + 1) * self.optimizer_batch_size
            states = {name: batch['states'][name][start:end] for name, state in self.state.items()}
            actions = {name: batch['actions'][name][start:end] for name, action in self.action.items()}
            rewards = batch['rewards'][start:end]
            terminals = batch['terminals'][start:end]
            internals = [batch['internals'][start:end] for internal in self.internal_inputs]
            minibatch = dict(states=states, actions=actions, rewards=rewards, terminals=terminals, internals=internals)

            # Last update, fetch return and diagnostics values for all samples
            feed_dict = {state: minibatch['states'][name] for name, state in self.state.items()}
            feed_dict.update({action: minibatch['actions'][name] for name, action in self.action.items()})
            feed_dict[self.reward] = minibatch['rewards']
            feed_dict[self.terminal] = minibatch['terminals']
            feed_dict.update({internal: minibatch['internals'][n] for n, internal in enumerate(self.internal_inputs)})

            fetches = (self.optimize, self.loss, self.loss_per_instance, self.kl_divergence, self.entropy)
            feed_dict.update(prev_distribution_tensors)
            _, loss, loss_per_instance, kl_divergence, entropy = self.session.run(fetches=fetches, feed_dict=feed_dict)

            losses.append(loss)
            losses_per_instance.append(loss_per_instance)
            kl_divergences.append(kl_divergence)

        loss_per_instance = np.concatenate(losses_per_instance)
        # FIXME should I sum these?
        loss = np.sum(losses)
        kl_divergence = np.sum(kl_divergences)

        self.logger.debug('Loss = {}'.format(loss))
        self.logger.debug('KL divergence = {}'.format(kl_divergence))
        self.logger.debug('Entropy = {}'.format(entropy))
        return loss, loss_per_instance
