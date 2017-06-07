from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest

from tensorforce import Configuration
from tensorforce.agents import NAFAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestNAFAgent(unittest.TestCase):

    def test_naf_agent(self):
        environment = MinimalTest(continuous=True)
        config = Configuration(
            batch_size=8,
            learning_rate=0.001,
            memory_capacity=800,
            first_update=80,
            repeat_update=4,
            target_update_frequency=20,
            states=environment.states,
            actions=environment.actions,
            exploration="OrnsteinUhlenbeckProcess",
            exploration_kwargs=dict(
                sigma=0.1,
                mu=0,
                theta=0.1
            )
        )
        network_builder = layered_network_builder(layers_config=[{'type': 'dense', 'size': 16},
                                                                 {'type': 'dense', 'size': 16}])
        agent = NAFAgent(config=config, network_builder=network_builder)
        runner = Runner(agent=agent, environment=environment)

        def episode_finished(r):
            return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

        runner.run(episodes=5000, episode_finished=episode_finished)
        print('NAF Agent: ' + str(runner.episode))
        self.assertTrue(runner.episode < 5000)
