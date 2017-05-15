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
from tensorforce.rl_agents.memory_agent import create_config

from tensorforce.agents import DQNAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestDQNAgent(unittest.TestCase):

    def test_dqn_agent(self):
        environment = MinimalTest(continuous=False)

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
        network_builder = layered_network_builder([{'type': 'dense', 'num_outputs': 16}, {'type': 'linear', 'num_outputs': 2}])
        agent = DQNAgent(config=config, network_builder=network_builder)

        runner = Runner(agent=agent, environment=environment)

        def episode_finished(r):
            return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

        runner.run(episodes=10000, episode_finished=episode_finished)
        self.assertTrue(runner.episode < 10000)