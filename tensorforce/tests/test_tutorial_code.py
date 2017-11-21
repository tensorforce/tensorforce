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


class TestTutorialCode(unittest.TestCase):
    """
    Validation of random code snippets as to be notified when old blog posts need to be changed.
    """

    class MyClient(object):

        def __init__(self, *args, **kwargs):
            pass

        def get_state(self):
            import numpy as np
            return np.random.rand(10)

        def execute(self, action):
            pass

    def test_reinforceio_homepage(self):
        """
        Code example from the homepage and README.md.
        """

        from tensorforce.agents import TRPOAgent

        # Create a Trust Region Policy Optimization agent
        agent = TRPOAgent(
            states_spec=dict(shape=(10,), type='float'),
            actions_spec=dict(type='int', num_actions=2),
            network_spec=[dict(type='dense', size=50), dict(type='dense', size=50)],
            batch_size=100,
        )

        # Get new data from somewhere, e.g. a client to a web app
        client = TestTutorialCode.MyClient('http://127.0.0.1', 8080)

        # Poll new state from client
        state = client.get_state()

        # Get prediction from agent, execute
        action = agent.act(states=state)
        reward = client.execute(action)

        # Add experience, agent automatically updates model according to batch size
        agent.observe(reward=reward, terminal=False)
        agent.close()

    def test_blogpost_introduction(self):
        """
        Test of introduction blog post examples.
        """
        import tensorflow as tf
        import numpy as np

        ### DQN agent example
        from tensorforce.agents import DQNAgent

        # Network is an ordered list of layers
        network_spec = [dict(type='dense', size=32), dict(type='dense', size=32)]

        # Define a state
        states = dict(shape=(10,), type='float')

        # Define an action
        actions = dict(type='int', num_actions=5)

        agent = DQNAgent(
            states_spec=states,
            actions_spec=actions,
            network_spec=network_spec,
            memory=dict(
                type='replay',
                capacity=1000
            ),
            batch_size=8,
            first_update=100,
            target_sync_frequency=10
        )

        agent.close()

        ### Code block: multiple states
        states = dict(
            image=dict(shape=(64, 64, 3), type='float'),
            caption=dict(shape=(20,), type='int')
        )

        # DQN does not support multiple states. Omit test for now.
        # agent = DQNAgent(config=config)

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
        network_spec = json.loads(network_json)

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
        network_spec = json.loads(modified_dense)

        ### Code block: Own layer type
        from tensorforce.core.networks import Layer

        class BatchNormalization(Layer):

            def __init__(self, variance_epsilon=1e-6, scope='batchnorm', summary_labels=None):
                super(BatchNormalization, self).__init__(scope=scope, summary_labels=summary_labels)
                self.variance_epsilon = variance_epsilon

            def tf_apply(self, x, update):
                mean, variance = tf.nn.moments(x, axes=tuple(range(x.shape.ndims - 1)))
                return tf.nn.batch_normalization(
                    x=x,
                    mean=mean,
                    variance=variance,
                    offset=None,
                    scale=None,
                    variance_epsilon=self.variance_epsilon
                )

        ### Test own layer

        states = dict(shape=(10,), type='float')
        network_spec = [
            {'type': 'dense', 'size': 32},
            {'type': BatchNormalization, 'variance_epsilon': 1e-9}
        ]

        agent = DQNAgent(
            states_spec=states,
            actions_spec=actions,
            network_spec=network_spec,
            memory=dict(
                type='replay',
                capacity=1000
            ),
            batch_size=8
        )

        agent.close()

        ### Code block: Own network builder
        from tensorforce.core.networks import Network

        class CustomNetwork(Network):

            def tf_apply(self, x, internals, update, return_internals=False):
                image = x['image']  # 64x64x3-dim, float
                caption = x['caption']  # 20-dim, int
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)

                # CNN
                weights = tf.get_variable(name='W1', shape=(3, 3, 3, 16), initializer=initializer)
                image = tf.nn.conv2d(image, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
                image = tf.nn.relu(image)
                image = tf.nn.max_pool(image, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

                weights = tf.get_variable(name='W2', shape=(3, 3, 16, 32), initializer=initializer)
                image = tf.nn.conv2d(image, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
                image = tf.nn.relu(image)
                image = tf.nn.max_pool(image, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

                image = tf.reshape(image, shape=(-1, 16 * 16, 32))
                image = tf.reduce_mean(image, axis=1)

                # LSTM
                weights = tf.get_variable(name='W3', shape=(30, 32), initializer=initializer)
                caption = tf.nn.embedding_lookup(params=weights, ids=caption)
                lstm = tf.contrib.rnn.LSTMCell(num_units=32)
                caption, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=caption, dtype=tf.float32)
                caption = tf.reduce_mean(caption, axis=1)

                # Combination
                if return_internals:
                    return tf.multiply(image, caption), list()
                else:
                    return tf.multiply(image, caption)

        ### Test own network builder

        states = dict(
            image=dict(shape=(64, 64, 3), type='float'),
            caption=dict(shape=(20,), type='int')
        )

        agent = DQNAgent(
            states_spec=states,
            actions_spec=actions,
            network_spec=CustomNetwork,
            memory=dict(
                type='replay',
                capacity=1000
            ),
            batch_size=8
        )

        agent.close()

        ### Code block: LSTM function
        from tensorforce.core.networks import Layer

        class Lstm(Layer):

            def __init__(self, size, scope='lstm', summary_labels=()):
                self.size = size
                super(Lstm, self).__init__(num_internals=1, scope=scope, summary_labels=summary_labels)

            def tf_apply(self, x, update, state):
                state = tf.contrib.rnn.LSTMStateTuple(c=state[:, 0, :], h=state[:, 1, :])
                self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.size)

                x, state = self.lstm_cell(inputs=x, state=state)

                internal_output = tf.stack(values=(state.c, state.h), axis=1)
                return x, (internal_output,)

            def internal_inputs(self):
                return super(Lstm, self).internal_inputs() + [tf.placeholder(dtype=tf.float32, shape=(None, 2, self.size))]

            def internal_inits(self):
                return super(Lstm, self).internal_inits() + [np.zeros(shape=(2, self.size))]

        ### Test LSTM
        states = dict(shape=(10,), type='float')
        network_spec = [
            {'type': 'flatten'},
            {'type': Lstm, 'size': 10}
        ]

        agent = DQNAgent(
            states_spec=states,
            actions_spec=actions,
            network_spec=network_spec,
            memory=dict(
                type='replay',
                capacity=1000
            ),
            batch_size=8
        )

        agent.close()

        ### Preprocessing configuration
        states = dict(shape=(84, 84, 3), type='float')
        preprocessing = [
            dict(
                type='image_resize',
                width=84,
                height=84
            ),
            dict(
                type='grayscale'
            ),
            dict(
                type='normalize'
            ),
            dict(
                type='sequence',
                length=4
            )
        ]

        ### Test preprocessing configuration

        agent = DQNAgent(
            states_spec=states,
            actions_spec=actions,
            network_spec=network_spec,
            memory=dict(
                type='replay',
                capacity=1000
            ),
            batch_size=8,
            first_update=100,
            target_sync_frequency=50,
            preprocessing=preprocessing
        )

        agent.close()

        ### Code block: Continuous action exploration

        exploration = dict(
            type='ornstein_uhlenbeck',
            sigma=0.1,
            mu=0,
            theta=0.1
        )

         ### Test continuous action exploration
        agent = DQNAgent(
            states_spec=states,
            actions_spec=actions,
            network_spec=network_spec,
            memory=dict(
                type='replay',
                capacity=1000
            ),
            batch_size=8,
            exploration=exploration
        )

        agent.close()

        ### Code block: Discrete action exploration

        exploration = dict(
            type='epsilon_decay',
            initial_epsilon=1.0,
            final_epsilon=0.01,
            timesteps=1e6
        )

        ### Test discrete action exploration
        agent = DQNAgent(
            states_spec=states,
            actions_spec=actions,
            network_spec=network_spec,
            memory=dict(
                type='replay',
                capacity=1000
            ),
            batch_size=8,
            exploration=exploration
        )

        agent.close()

    def test_blogpost_introduction_runner(self):
        from tensorforce.environments.minimal_test import MinimalTest
        from tensorforce.agents import DQNAgent
        from tensorforce.execution import Runner

        environment = MinimalTest(specification=[('int', ())])

        network_spec = [
            dict(type='dense', size=32)
        ]

        agent = DQNAgent(
            states_spec=environment.states,
            actions_spec=environment.actions,
            network_spec=network_spec,
            memory=dict(
                type='replay',
                capacity=1000
            ),
            batch_size=8,
            first_update=100,
            target_sync_frequency=50
        )
        runner = Runner(agent=agent, environment=environment)

        def episode_finished(runner):
            if runner.episode % 100 == 0:
                print(sum(runner.episode_rewards[-100:]) / 100)
            return runner.episode < 100 \
                or not all(reward >= 1.0 for reward in runner.episode_rewards[-100:])

        # runner.run(episodes=1000, episode_finished=episode_finished)
        runner.run(episodes=10, episode_finished=episode_finished)  # Only 10 episodes for this test

        ### Code block: next
        agent = DQNAgent(
            states_spec=environment.states,
            actions_spec=environment.actions,
            network_spec=network_spec,
            memory=dict(
                type='replay',
                capacity=1000
            ),
            batch_size=8,
            first_update=100,
            target_sync_frequency=50
        )

        # max_episodes = 1000
        max_episodes = 10  # Only 10 episodes for this test
        max_timesteps = 2000

        episode = 0
        episode_rewards = list()

        while True:
            state = environment.reset()
            agent.reset()

            timestep = 0
            episode_reward = 0
            while True:
                action = agent.act(states=state)
                state, terminal, reward = environment.execute(actions=action)
                agent.observe(terminal=terminal, reward=reward)

                timestep += 1
                episode_reward += reward

                if terminal or timestep == max_timesteps:
                    break

            episode += 1
            episode_rewards.append(episode_reward)

            if all(reward >= 1.0 for reward in episode_rewards[-100:]) or episode == max_episodes:
                break

        agent.close()
        environment.close()
