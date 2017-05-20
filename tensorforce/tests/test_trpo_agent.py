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

from tensorforce.agents import TRPOAgent
from tensorforce.config import create_config
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestTRPOAgent(unittest.TestCase):
    def test_discrete(self):
        environment = MinimalTest(continuous=False)

        config = {
            'batch_size': 16,
            "override_line_search": False,
            "cg_iterations": 20,
            "use_gae": False,
            "normalize_advantage": False,
            "gae_lambda": 0.97,
            "cg_damping": 0.001,
            "line_search_steps": 20,
            'max_kl_divergence': 0.05,
            'max_episode_length': 4,
            'continuous': False,
            'state_shape': (2,),
            'actions': 2,
            'gamma': 0.99
        }

        config = create_config(config)
        tf.reset_default_graph()

        network_builder = layered_network_builder([{'type': 'dense',
                                                                 'num_outputs': 8}])
        agent = TRPOAgent(config=config, network_builder=network_builder)
        runner = Runner(agent=agent, environment=environment)

        def episode_finished(r):
            return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

        runner.run(episodes=10000, episode_finished=episode_finished)
        self.assertTrue(runner.episode < 10000)

    def test_trpo_agent_continuous(self):
        environment = MinimalTest(continuous=True)

        config = {
            'batch_size': 16,
            "override_line_search": False,
            "cg_iterations": 20,
            "use_gae": False,
            "normalize_advantage": False,
            "gae_lambda": 0.97,
            "cg_damping": 0.001,
            "line_search_steps": 20,
            'max_kl_divergence': 0.05,
            'max_episode_length': 4,
            'continuous': True,
            'state_shape': (2,),
            'actions': 2,
            'gamma': 0.99
        }

        config = create_config(config)
        tf.reset_default_graph()

        network_builder = layered_network_builder([{'type': 'dense',
                                                    'num_outputs': 8}])
        agent = TRPOAgent(config=config, network_builder=network_builder)
        runner = Runner(agent=agent, environment=environment)

        def episode_finished(r):
            return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

        runner.run(episodes=10000, episode_finished=episode_finished)
        self.assertTrue(runner.episode < 10000)