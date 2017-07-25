from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest
from six.moves import xrange

from tensorforce import Configuration
from tensorforce.agents import NAFAgent
from tensorforce.core.networks import layered_network_builder, layers
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestNAFAgent(unittest.TestCase):

    def test_naf_agent(self):

        passed = 0
        for _ in xrange(5):
            environment = MinimalTest(definition=True)
            config = Configuration(
                batch_size=8,
                learning_rate=0.001,
                exploration=dict(
                    type='ornstein_uhlenbeck'
                ),
                memory_capacity=800,
                first_update=80,
                target_update_frequency=20,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([dict(type='dense', size=32)])
                # batch_size=8,
                # learning_rate=0.0025,
                # # exploration="OrnsteinUhlenbeckProcess",
                # # exploration_kwargs=dict(
                # #     sigma=0.1,
                # #     mu=0,
                # #     theta=0.1
                # # ),
                # discount=0.99,
                # memory_capacity=800,
                # first_update=80,
                # repeat_update=4,
                # target_update_frequency=20,
                # states=environment.states,
                # actions=environment.actions,
                # clip_gradients=5.0,
                # network=layered_network_builder([dict(type='dense', size=32), dict(type='dense', size=32)])
            )
            agent = NAFAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=1500, episode_finished=episode_finished)
            print('NAF agent: ' + str(runner.episode))
            if runner.episode < 1500:
                passed += 1

        print('NAF agent passed = {}'.format(passed))
        self.assertTrue(passed >= 3)

    def test_multi(self):
        passed = 0

        def network_builder(inputs):
            state0 = layers['dense'](x=inputs['state0'], size=32)
            state1 = layers['dense'](x=inputs['state1'], size=32)
            state2 = layers['dense'](x=inputs['state2'], size=32)
            return state0 * state1 * state2

        for _ in xrange(5):
            environment = MinimalTest(definition=[True, (True, 2), (True, (1, 2))])
            config = Configuration(
                batch_size=8,
                learning_rate=0.001,
                exploration=dict(
                    type='ornstein_uhlenbeck'
                ),
                memory_capacity=800,
                first_update=80,
                target_update_frequency=20,
                states=environment.states,
                actions=environment.actions,
                network=network_builder
            )
            agent = NAFAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 10 or not all(x >= 1.0 for x in r.episode_rewards[-10:])

            runner.run(episodes=2000, episode_finished=episode_finished)
            print('NAF agent (multi-state/action): ' + str(runner.episode))
            if runner.episode < 2000:
                passed += 1

        print('NAF agent (multi-state/action) passed = {}'.format(passed))
        self.assertTrue(passed >= 4)
