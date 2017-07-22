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
from mock import Mock


class TestTutorialCode(unittest.TestCase):

    def test_reinforceio_homepage(self):
        """
        Code example from the homepage and README.md.
        """
        MyClient = Mock()

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

    def test_introduction_dqnagent(self):
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
