# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from datetime import datetime
import logging
import unittest
import sys

from test.unittest_environment import UnittestEnvironment


logging.getLogger('tensorflow').disabled = True


class TestReadme(unittest.TestCase):

    def test_readme(self):
        sys.stdout.write('\n{} {}: '.format(
            datetime.now().strftime('%H:%M:%S'), self.__class__.__name__[4:]
        ))

        environment = UnittestEnvironment(
            states=dict(type='float', shape=(10,)),
            actions=dict(type='int', shape=(), num_values=5),
            timestep_range=(1, 5)
        )

        def get_current_state():
            return environment.reset()

        def execute_decision(x):
            return environment.execute(actions=x)[2]

        # ==========

        from tensorforce.agents import PolicyAgent

        # Instantiate a Tensorforce agent
        agent = PolicyAgent(
            states=dict(type='float', shape=(10,)),
            actions=dict(type='int', num_values=5),
            memory=10000,
            update=dict(unit='timesteps', batch_size=64),
            optimizer=dict(type='adam', learning_rate=3e-4),
            network='auto',
            objective='policy_gradient',
            reward_estimation=dict(horizon=20)
        )

        # Initialize the agent
        agent.initialize()

        # Retrieve the latest (observable) environment state
        state = get_current_state()  # (float array of shape [10])

        # Query the agent for its action decision
        action = agent.act(states=state)  # (scalar between 0 and 4)

        # Execute the decision and retrieve the current performance score
        reward = execute_decision(action)  # (any scalar float)

        # Pass feedback about performance (and termination) to the agent
        agent.observe(reward=reward, terminal=False)

        # ==========

        agent.close()
        environment.close()
        self.assertTrue(expr=True)

        sys.stdout.write('.')
        sys.stdout.flush()
