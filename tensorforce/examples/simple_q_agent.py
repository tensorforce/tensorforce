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

import numpy as np
import tensorflow as tf
from six.moves import xrange

from tensorforce.agents import MemoryAgent
from tensorforce.models import Model
from tensorforce.models.neural_networks import NeuralNetwork

from tensorforce.config import Config
from tensorforce.external.openai_gym import OpenAIGymEnvironment
from tensorforce.execution import Runner


class SimpleQModel(Model):
    # Default config values
    default_config = {
        "gamma": 0.9,
        "network_layers": [{
            "type": "dense",
            "num_outputs": 16
        }]
    }

    def __init__(self, config, scope):
        super(SimpleQModel, self).__init__(config, scope)
        self.action_count = self.config.actions

        self.random = np.random.RandomState()

        with tf.device(self.config.tf_device):
            # Create state placeholder
            # self.batch_shape is [None] (set in Model.__init__)
            self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")

            # Create neural network
            output_layer = [{"type": "dense", "num_outputs": self.action_count}]
            self.network = NeuralNetwork(self.config.network_layers + output_layer, self.state, scope=self.scope + "network")
            self.network_out = self.network.get_output()

            # Create operations
            self.create_ops()
            self.init_op = tf.global_variables_initializer()

            # Create optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.alpha)

    def get_action(self, state, episode=1):
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
        # TODO: fix. https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.tjk7nwi6a
        # Get Q values for current states
        current_q = self.session.run(self.network_out, {
            self.state: batch['states']
        })

        # Get Q values for next states
        next_q = self.session.run(self.network_out, {
            self.state: batch['next_states']
        })

        # Bellmann equation Q = r + y * Q'
        current_q[0,'actions'] = batch['rewards'] + (1. - batch['terminals'].astype(float)) \
                                       * self.config.gamma * np.max(next_q)

        self.session.run(self.optimize_op, {
            self.state: batch['next_states'],
            self.q_targets: q_targets
        })

    def initialize(self):
        self.session.run(self.init_op)

    def create_ops(self):
        with tf.name_scope(self.scope):
            with tf.name_scope("predict"):
                self.q_action = tf.argmax(self.network_out, axis=1)

            with tf.name_scope("update"):
                self.q_targets = tf.placeholder(tf.float32, [None, self.action_count], name='q_targets')
                self.loss = tf.reduce_sum(tf.square(self.q_targets - self.network_out))
                self.optimize_op = self.optimizer.minimize(self.loss)


class SimpleQAgent(MemoryAgent):
    """
    Simple agent extending MemoryAgent
    """
    name = 'SimpleQAgent'

    model_ref = SimpleQModel

    default_config = {
        "memory_capacity": 100,
        "batch_size": 10,
        "update_rate": 0.5,
        "update_repeat": 1
    }


def main():
    gym_id = 'LunarLander-v2'
    max_episodes = 100
    max_timesteps = 100

    env = OpenAIGymEnvironment(gym_id, monitor=False, monitor_video=False)

    config = Config({
        'repeat_actions': 1,
        'actions': env.actions,
        'action_shape': env.action_shape,
        'state_shape': env.state_shape
    })

    agent = SimpleQAgent(config, "simpleq")

    runner = Runner(agent, env)

    def episode_finished(r):
        if r.episode % 1 == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 10 rewards: {}".format(np.mean(r.episode_rewards[-10:])))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))
    runner.run(max_episodes, max_timesteps, episode_finished=episode_finished)
    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode + 1))

if __name__ == '__main__':
    main()
