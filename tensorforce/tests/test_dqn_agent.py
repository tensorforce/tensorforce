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
import tensorflow as tf

import unittest
from six.moves import xrange

from tensorforce.config import create_config
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.agents import DQNAgent

class TestDQNAgent(unittest.TestCase):

    def test_dqn_agent(self):
        config = {
            'seed': 10,
            'batch_size': 16,
            'state_shape': (2,),
            'actions': 2,
            'action_shape': (),
            'update_rate': 1,
            'update_repeat': 4,
            'min_replay_size': 50,
            'memory_capacity': 50,
            "exploration": "epsilon_decay",
            "exploration_param": {
                "epsilon": 1,
                "epsilon_final": 0,
                "epsilon_states": 50
            },
            'target_network_update_rate': 1.0 ,
            'use_target_network': True,
            "alpha": 0.0005,
            "gamma": 0.99,
            "tau": 1.0
        }

        tf.reset_default_graph()
        tf.set_random_seed(10)

        config = create_config(config)
        network_builder = NeuralNetwork.layered_network(layers=[{'type': 'dense', 'num_outputs': 16}, {'type': 'linear', 'num_outputs': 2}])
        agent = DQNAgent(config=config, network_builder=network_builder)

        state = (1, 0)
        rewards = [0.0] * 100
        for n in xrange(10000):
            action = agent.get_action(state=state)
            if action == 0:
                state = (1, 0)
                reward = 0.0
                terminal = False
            else:
                state = (0, 1)
                reward = 1.0
                terminal = False
            agent.add_observation(state=state, action=action, reward=reward, terminal=terminal)
            rewards[n % 100] = reward

            if sum(rewards) == 100.0:
                return

        assert(sum(rewards) == 100.0)
