from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest
from six.moves import xrange

from tensorforce import Configuration
from tensorforce.agents import NAFAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestNAFAgent(unittest.TestCase):

    def test_naf_agent(self):

        passed = 0
        for _ in xrange(10):
            environment = MinimalTest(continuous=True)
            config = Configuration(
                batch_size=8,
                learning_rate=0.0025,
                # exploration="OrnsteinUhlenbeckProcess",
                # exploration_kwargs=dict(
                #     sigma=0.1,
                #     mu=0,
                #     theta=0.1
                # ),
                discount=0.99,
                memory_capacity=800,
                first_update=80,
                repeat_update=4,
                target_update_frequency=20,
                states=environment.states,
                actions=environment.actions,
                clip_gradients=5.0,
                network=layered_network_builder([dict(type='dense', size=32), dict(type='dense', size=32)])
            )
            agent = NAFAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=10000, episode_finished=episode_finished)
            # print('NAF Agent: ' + str(runner.episode))
            if runner.episode < 10000:
                passed += 1
                print('passed')
            else:
                print('failed')

        print('NAF Agent passed = {}'.format(passed))
        self.assertTrue(passed >= 8)
