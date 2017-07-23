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
Test for examples from the reinforce.io website, blogposts and other examples.
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest


class TestTutorialCode(unittest.TestCase):

    def test_reinforceio_homepage(self):
        """
        Code example from the homepage and README.md.
        """
        class MyClient(object):
            def __init__(self, *args, **kwargs):
                pass

            def get_state(self):
                import numpy as np
                return np.random.rand(10)

            def execute(self, action):
                pass

        from tensorforce import Configuration
        from tensorforce.agents import TRPOAgent
        from tensorforce.core.networks import layered_network_builder

        config = Configuration(
            batch_size=100,
            states=dict(shape=(10,), type='float'),
            actions=dict(continuous=False, num_actions=2),
            network=layered_network_builder([dict(type='dense', size=50), dict(type='dense', size=50)])
        )

        # Create a Trust Region Policy Optimization agent
        agent = TRPOAgent(config=config)

        # Get new data from somewhere, e.g. a client to a web app
        client = MyClient('http://127.0.0.1', 8080)

        # Poll new state from client
        state = client.get_state()

        # Get prediction from agent, execute
        action = agent.act(state=state)
        reward = client.execute(action)

        # Add experience, agent automatically updates model according to batch size
        agent.observe(reward=reward, terminal=False)

    def test_blogpost_introduction(self):
        """
        Test of introduction blog post examples.
        """
        import tensorflow as tf
        import numpy as np

        ### DQN agent example

        from tensorforce import Configuration
        from tensorforce.agents import DQNAgent
        from tensorforce.core.networks import layered_network_builder

        # Define a network builder from an ordered list of layers
        layers = [dict(type='dense', size=32),
                  dict(type='dense', size=32)]
        network = layered_network_builder(layers_config=layers)

        # Define a state
        states = dict(shape=(10,), type='float')

        # Define an action (models internally assert whether
        # they support continuous and/or discrete control)
        actions = dict(continuous=False, num_actions=5)

        # The agent is configured with a single configuration object
        agent_config = Configuration(
            batch_size=8,
            learning_rate=0.001,
            memory_capacity=800,
            first_update=80,
            repeat_update=4,
            target_update_frequency=20,
            states=states,
            actions=actions,
            network=network
        )
        agent = DQNAgent(config=agent_config)

        ### Code block: multiple states

        states = dict(
            image=dict(shape=(64, 64, 3), type='float'),
            caption=dict(shape=(20,), type='int')
        )

        agent_config.states = states
        # DQN does not support multiple states. Omit test for now.
        # agent = DQNAgent(config=agent_config)


        ### Code block: DQN observer function

        def observe(self, reward, terminal):
            super(DQNAgent, self).observe(reward, terminal)
            if self.timestep >= self.first_update \
                    and self.timestep % self.target_update_frequency == 0:
                self.model.update_target()

        ### Code block: Network config JSON

        network_json = """
        [
            {
                "type": "conv2d",
                "size": 32,
                "window": 8,
                "stride": 4
            },
            {
                "type": "conv2d",
                "size": 64,
                "window": 4,
                "stride": 2
            },
            {
                "type": "flatten"
            },
            {
                "type": "dense",
                "size": 512
            }
        ]
        """

        ### Test json

        import json
        network_config = json.loads(network_json)
        network = layered_network_builder(network_config)

        ### Test from_json import

        from tensorforce.core.networks import from_json

        ### Code block: Modified dense layer

        modified_dense = """
        [
            {
                "type": "dense",
                "size": 64,
                "bias": false,
                "activation": "selu",
                "l2_regularization": 0.001
            }
        ]
        """

        ### Test json

        network_config = json.loads(network_json)
        network = layered_network_builder(network_config)

        ### Code block: Own layer type

        def batch_normalization(x, variance_epsilon=1e-6):
            mean, variance = tf.nn.moments(x, axes=tuple(range(x.shape.ndims - 1)))
            x = tf.nn.batch_normalization(x, mean=mean, variance=variance,
                                          variance_epsilon=variance_epsilon)
            return x

        ### Test own layer

        network_config = [{"type": batch_normalization,
                           "variance_epsilon": 1e-9}]
        network = layered_network_builder(network_config)

        ### Code block: Own network builder

        def network_builder(inputs):
            image = inputs['image']  # 64x64x3-dim, float
            caption = inputs['caption']  # 20-dim, int

            with tf.variable_scope('cnn'):
                weights = tf.Variable(tf.random_normal(shape=(3, 3, 3, 16), stddev=0.01))
                image = tf.nn.conv2d(image, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
                image = tf.nn.relu(image)
                image = tf.nn.max_pool(image, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

                weights = tf.Variable(tf.random_normal(shape=(3, 3, 16, 32), stddev=0.01))
                image = tf.nn.conv2d(image, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
                image = tf.nn.relu(image)
                image = tf.nn.max_pool(image, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

                image = tf.reshape(image, shape=(-1, 16 * 16, 32))
                image = tf.reduce_mean(image, axis=1)

            with tf.variable_scope('lstm'):
                weights = tf.Variable(tf.random_normal(shape=(30, 32), stddev=0.01))
                caption = tf.nn.embedding_lookup(params=weights, ids=caption)
                lstm = tf.contrib.rnn.LSTMCell(num_units=32)
                caption, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=caption, dtype=tf.float32)
                caption = tf.reduce_mean(caption, axis=1)

            return tf.multiply(image, caption)

        ### Test own network builder

        agent_config.network=network_builder
        agent = DQNAgent(config=agent_config)

        ### Code block: LSTM function

        def lstm(x):
            size = x.get_shape()[1].value
            internal_input = tf.placeholder(dtype=tf.float32, shape=(None, 2, size))
            lstm = tf.contrib.rnn.LSTMCell(num_units=size)
            state = tf.contrib.rnn.LSTMStateTuple(internal_input[:, 0, :],
                                                  internal_input[:, 1, :])
            x, state = lstm(inputs=x, state=state)
            internal_output = tf.stack(values=(state.c, state.h), axis=1)
            internal_init = np.zeros(shape=(2, size))
            return x, [internal_input], [internal_output], [internal_init]

        ### Test LSTM

        network_config = [{"type": lstm}]
        network = layered_network_builder(network_config)