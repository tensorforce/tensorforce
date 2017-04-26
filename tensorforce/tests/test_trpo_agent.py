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

from tensorforce.config import create_config
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.agents import TRPOAgent


class TestTRPOAgent(unittest.TestCase):

    def test_trpo_agent(self):
        config = {
            'batch_size': 8,
            'max_kl_divergence': 0.01,
            'max_episode_length': 4,
            'continuous': False,
            'state_shape': (2,),
            'actions': 2}

        config = create_config(config)
        network_builder = NeuralNetwork.layered_network(layers=[{'type': 'dense', 'num_outputs': 32}])
        agent = TRPOAgent(config=config, network_builder=network_builder)

        state = (1, 0)
        rewards = [0.0] * 100
        for n in range(100):
            action = agent.get_action(state=state)
            if action == 0:
                state = (1, 0)
                reward = 0.0
                terminal = False
            else:
                state = (0, 1)
                reward = 1.0
                terminal = True
            agent.add_observation(state=state, action=action, reward=reward, terminal=terminal)
            rewards[n % 100] = reward
            if sum(rewards) == 100.0:
                return
        # self.assertTrue(False)
