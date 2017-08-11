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
# ==============================================================================

"""
Deep Q network using Nstep rewards as desribed in Asynchronous Methods for Reinforcement Learning
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorforce import util
from tensorforce.models import DQNModel


class DQNNstepModel(DQNModel):
    def create_tf_operations(self, config):
        # create a nstep reward placeholder for each action
        with tf.variable_scope('placeholder'):
            self.nstep_rewards = dict()
            for name, state in config.actions.items():
                self.nstep_rewards[name] = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward-{}'.format(name))

        super(DQNNstepModel, self).create_tf_operations(config)

    def create_q_deltas(self, config):
        """
        Nstep rewards are calculated in update_feed_dict and passed into self.nstep_rewards
        So we just calculate the delta between training output and rewards
        """
        deltas = list()
        for name, action in self.action.items():
            reward = self.nstep_rewards[name]
            for _ in range(len(config.actions[name].shape)):
                reward = tf.expand_dims(input=reward, axis=1)
            delta = reward - self.q_values[name]
            delta = tf.reshape(tensor=delta, shape=(-1, util.prod(config.actions[name].shape)))
            deltas.append(delta)
        return deltas

    def update_feed_dict(self, batch):
        # assume temporally consistent sequence
        # get state time + 1
        feed_dict = {next_state: [batch['states'][name][-1]] for name, next_state in self.next_state.items()}
        feed_dict.update({internal: [batch['internals'][n][-1]] for n, internal in enumerate(self.internal_inputs)})
        # calcualte nstep rewards
        target_q_vals = self.session.run(self.target_values, feed_dict=feed_dict)
        nstep_rewards = dict()

        for name, value in target_q_vals.items():
            nstep_rewards[name] = util.cumulative_discount(rewards=batch['rewards'][:-1], terminals=batch['terminals'][:-1],
                                                           discount=self.discount, cumulative_start=target_q_vals[name][0])

        # create update feed dict
        feed_dict = {state: batch['states'][name][:-1] for name, state in self.state.items()}
        feed_dict.update({action: batch['actions'][name][:-1] for name, action in self.action.items()})
        feed_dict.update({internal: [batch['internals'][n][:-1]] for n, internal in enumerate(self.internal_inputs)})
        feed_dict.update({self.nstep_rewards[name]: nstep_rewards[name] for name, reward in nstep_rewards.items()})
        return feed_dict
