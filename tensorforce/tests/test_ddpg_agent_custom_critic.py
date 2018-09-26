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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest

from tensorforce.tests.base_agent_test import BaseAgentTest
from tensorforce.agents import DDPGAgent
from tensorforce.core.networks import Network


class Critic(Network):

    def tf_apply(self, x, internals, update, return_internals=False):
        import tensorflow as tf
        image = x['states']
        action = x['actions']
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)

        # CNN
        weights = tf.get_variable(name='W1', shape=(3, 3, 3, 16), initializer=initializer)
        out = tf.nn.conv2d(image, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
        out = tf.nn.relu(out)
        out = tf.nn.max_pool(out, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        out = tf.layers.flatten(out)
        # append action
        out = tf.concat([out, action], axis=-1)
        out = tf.layers.dense(out)

        if return_internals:
            return out, None
        else:
            return out


class TestDDPGAgentCustomCritic(BaseAgentTest, unittest.TestCase):

    agent = DDPGAgent
    config = dict(
        update_mode=dict(
            unit='timesteps',
            batch_size=8,
            frequency=8
        ),
        memory=dict(
            type='replay',
            include_next_states=True,
            capacity=100
        ),
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        critic_network=Critic,
        target_sync_frequency=10
    )
    exclude_multi = True


