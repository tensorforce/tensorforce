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
Simple Q agent as an example on how to implement own agents and models.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorforce.agents import MemoryAgent
from tensorforce.models import Model
from tensorforce.models.neural_networks import NeuralNetwork

from tensorforce.config import Config
from tensorforce.external.openai_gym import OpenAIGymEnvironment
from tensorforce.execution import Runner


class SimpleQModel(Model):
    # Default config values
    default_config = {
        "alpha": 0.01,
        "gamma": 0.99,
        "network_layers": [{
            "type": "linear",
            "num_outputs": 16
        }]
    }

    def __init__(self, config, scope):
        """
        Initialize model, build network and tensorflow ops

        :param config: Config object or dict
        :param scope: tensorflow scope name
        """
        super(SimpleQModel, self).__init__(config, scope)
        self.action_count = self.config.actions

        self.random = np.random.RandomState()

        with tf.device(self.config.tf_device):
            # Create state placeholder
            # self.batch_shape is [None] (set in Model.__init__)
            self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")

            # Create neural network
            output_layer = [{"type": "linear", "num_outputs": self.action_count}]
            self.network = NeuralNetwork(self.config.network_layers + output_layer, self.state, scope=self.scope + "network")
            self.network_out = self.network.get_output()

            # Create operations
            self.create_ops()
            self.init_op = tf.global_variables_initializer()

            # Create optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.alpha)

    def get_action(self, state, episode=1):
        """
        Get action for a given state

        :param state: ndarray containing the state
        :param episode: number of episode (for epsilon decay and alike)
        :return: action
        """

        # self.exploration is initialized in Model.__init__ and provides an API for different explorations methods,
        # such as epsilon greedy.
        epsilon = self.exploration(episode, self.total_states)  # returns a float

        if self.random.random_sample() < epsilon:
            action = self.random.randint(0, self.action_count)
        else:
            action = self.session.run(self.q_action, {
                self.state: [state]
            })[0]

        self.total_states += 1
        return action

    def update(self, batch):
        """
        Update model parameters

        :param batch: memory batch
        :return:
        """
        # Get Q values for next states
        next_q = self.session.run(self.network_out, {
            self.state: batch['next_states']
        })

        # Bellmann equation Q = r + y * Q'
        q_targets = batch['rewards'] + (1. - batch['terminals'].astype(float)) \
                                       * self.config.gamma * np.max(next_q, axis=1)

        self.session.run(self.optimize_op, {
            self.state: batch['states'],
            self.actions: batch['actions'],
            self.q_targets: q_targets
        })

    def initialize(self):
        """
        Initialize model variables
        :return:
        """
        self.session.run(self.init_op)

    def create_ops(self):
        """
        Create tensorflow ops

        :return:
        """
        with tf.name_scope(self.scope):
            with tf.name_scope("predict"):
                self.q_action = tf.argmax(self.network_out, axis=1)

            with tf.name_scope("update"):
                # These are the target Q values, i.e. the actual rewards plus the expected values of the next states
                # (Bellman equation).
                self.q_targets = tf.placeholder(tf.float32, [None], name='q_targets')

                # Actions that have been taken.
                self.actions = tf.placeholder(tf.int32, [None], name='actions')

                # We need the Q values of the current states to calculate the difference ("loss") between the
                # expected values and the new values (q targets). Therefore we do a forward-pass
                # and reduce the results to the actions that have been taken.

                # One_hot tensor of the actions that have been taken.
                actions_one_hot = tf.one_hot(self.actions, self.action_count, 1.0, 0.0, name='action_one_hot')

                # Training output, reduced to the actions that have been taken.
                q_values_actions_taken = tf.reduce_sum(self.network_out * actions_one_hot, axis=1,
                                                       name='q_acted')

                # The loss is the difference between the q_targets and the expected q values.
                self.loss = tf.reduce_sum(tf.square(self.q_targets - q_values_actions_taken))
                self.optimize_op = self.optimizer.minimize(self.loss)


class SimpleQAgent(MemoryAgent):
    """
    Simple agent extending MemoryAgent
    """
    name = 'SimpleQAgent'

    model_ref = SimpleQModel

    default_config = {
        "memory_capacity": 1000,  # hold the last 100 observations in the replay memory
        "batch_size": 10,  # train model with batches of 10
        "update_rate": 0.5,  # update parameters every other step
        "update_repeat": 1  # repeat update only one time
    }


def main():
    gym_id = 'CartPole-v0'
    max_episodes = 10000
    max_timesteps = 1000

    env = OpenAIGymEnvironment(gym_id, monitor=False, monitor_video=False)

    config = Config({
        'repeat_actions': 1,
        'actions': env.actions,
        'action_shape': env.action_shape,
        'state_shape': env.state_shape,
        'exploration': 'constant',
        'exploration_args': [0.1]
    })

    agent = SimpleQAgent(config, "simpleq")

    runner = Runner(agent, env)

    def episode_finished(r):
        if r.episode % 10 == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 10 rewards: {}".format(np.mean(r.episode_rewards[-10:])))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))
    runner.run(max_episodes, max_timesteps, episode_finished=episode_finished)
    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode + 1))

if __name__ == '__main__':
    main()
