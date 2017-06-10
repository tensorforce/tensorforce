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
Simple Q agent as an example on how to implement new agents and models.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import logging

from tensorforce.core import MemoryAgent, Model
from tensorforce.core.networks import NeuralNetwork, layers

from tensorforce.config import Configuration
from tensorforce.environments.openai_gym import OpenAIGym
from tensorforce.execution import Runner

from tensorforce.agents import create_agent


class SimpleQModel(Model):
    # Default config values
    default_config = {
        "alpha": 0.01,
        "gamma": 0.99
    }

    allows_continuous_actions = False
    allows_discrete_actions = True

    def __init__(self, config):
        """
        Initialize model, build network and tensorflow ops

        :param config: Config object or dict
        :param scope: tensorflow scope name
        """
        config.default(SimpleQModel.default_config)
        self.action_count = config.actions['action'].num_actions
        self.gamma = config.gamma

        self.random = np.random.RandomState()

        # Create optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.alpha)

        super(SimpleQModel, self).__init__(config)

    def get_action(self, state, internals=None):
        """
        Get action for a given state

        :param state: ndarray containing the state
        :param episode: number of episode (for epsilon decay and alike)
        :return: action
        """
        action = self.session.run(self.q_action, {
            self.state['state']: [state['state']]
        })
        return dict(action=action[0]), internals

    def update(self, batch):
        """


        :param batch: replay_memory batch
        :return:
        """
        self.session.run(self.optimize, {
            self.state['state']: batch['states']['state'],
            self.action['action']: batch['actions']['action'],
            self.reward: batch['rewards'],
            self.terminal: batch['terminals']
        })


    def create_tf_operations(self, config):
        """
        Create tensorflow ops

        :return:
        """
        super(SimpleQModel, self).create_tf_operations(config)

        with tf.name_scope("simpleq"):

            self.network = NeuralNetwork(config.network, inputs=self.state)
            self.network_output = layers['linear'](x=self.network.output, size=self.action_count)

            with tf.name_scope("predict"):
                self.q_action = tf.argmax(self.network_output, axis=1)

            with tf.name_scope("update"):
                # We need the Q values of the current states to calculate the difference ("loss") between the
                # expected values and the new values (q targets). Therefore we do a forward-pass
                # and reduce the results to the actions that have been taken.

                # One_hot tensor of the actions that have been taken.
                actions_one_hot = tf.one_hot(self.action['action'][:-1], self.action_count, 1.0, 0.0, name='action_one_hot')

                # Training output, reduced to the actions that have been taken.
                q_values_actions_taken = tf.reduce_sum(self.network_output[:-1] * actions_one_hot, axis=1,
                                                       name='q_acted')

                # Expected values for the next states
                q_output = tf.reduce_max(self.network_output[1:], axis=1, name='q_expected')

                # Bellmann equation Q = r + y * Q'
                q_targets = self.reward[:-1] + (1. - tf.cast(self.terminal[:-1], tf.float32)) \
                                               * self.gamma * q_output

                # The loss is the difference between the q_targets and the expected q values.
                self.loss = tf.reduce_sum(tf.square(q_targets - q_values_actions_taken))
                # self.optimize_op = self.optimizer.minimize(self.loss)

                tf.losses.add_loss(self.loss)


class SimpleQAgent(MemoryAgent):
    """
    Simple agent extending MemoryAgent
    """
    name = 'SimpleQAgent'

    model = SimpleQModel

    default_config = {
        "memory_capacity": 1000,  # hold the last 100 observations in the replay memory
        "batch_size": 10,  # train model with batches of 10
        "update_frequency": 2,  # update parameters every other step
        "update_repeat": 1,  # repeat update only one time
        "min_replay_size": 0 # minimum size of replay memory before updating
    }


def main():
    gym_id = 'CartPole-v0'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    max_episodes = 10000
    max_timesteps = 1000

    env = OpenAIGym(gym_id, monitor=False, monitor_video=False)

    config = Configuration(
        repeat_actions=1,
        actions=env.actions,
        states=env.states,
        exploration='constant',
        exploration_args=[0.1],
        network=[{"type": "linear", "size": 16}]
    )

    agent = create_agent(SimpleQAgent, config)

    runner = Runner(agent, env)

    def episode_finished(r):
        if r.episode % 10 == 0:
            logger.info("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 10 rewards: {}".format(np.mean(r.episode_rewards[-10:])))
        return True

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))
    runner.run(max_episodes, max_timesteps, episode_finished=episode_finished)
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode + 1))

if __name__ == '__main__':
    main()
