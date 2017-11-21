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
import numpy as np
from tensorforce.agents import VPGAgent
from tensorforce.core.baselines import Baseline


class TestRewardEstimation(unittest.TestCase):
    #TODO - reward is now computed using TensorFlow operations and not easily
    # accessible. Need to rethink how to test things like this.

    def test_basic(self):

        kwargs = dict(
            discount=0.75,
            batch_size=8,
            learning_rate=0.001,
        )
        # agent = VPGAgent(
        #     states_spec=dict(shape=(1,)),
        #     actions_spec=dict(type='int', num_actions=2),
        #     network_spec=[dict(type='dense', size=32)],
        #     config=config
        # )

        states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        actions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        rewards = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        terminals = [False, False, False, False, True, False, False, False, True]
        discounted_rewards = np.array([
            0.75 + 0.75 ** 4, 1.0 + 0.75 ** 3, 0.75 ** 2, 0.75, 1.0,
            1.0 + 0.75 ** 2, 0.75, 1.0, 0.0
        ])

        feed_dict=dict()
        # feed_dict[agent.model.reward_input] = rewards
        # fetches = [agent.model.get_reward()]
        # result = agent.model.session.run(feed_dict=feed_dict, fetches=fetches)


        expected = discounted_rewards

       #self.assertTrue((result == expected).all())

    def test_baseline(self):
        kwargs = dict(
            discount=0.75,
            batch_size=8,
            learning_rate=0.001,
        )
        # agent = VPGAgent(
        #     states_spec=dict(shape=(1,)),
        #     actions_spec=dict(type='int', num_actions=2),
        #     network_spec=[dict(type='dense', size=32)],
        #     config=config
        # )

        states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rewards = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        terminals = [False, False, False, False, True, False, False, False, True]
        discounted_rewards = np.array([
            0.75 + 0.75 ** 4, 1.0 + 0.75 ** 3, 0.75 ** 2, 0.75, 1.0,
            1.0 + 0.75 ** 2, 0.75, 1.0, 0.0
        ])
        baseline = np.array([0.25, 0.5, 0.0, 0.25, 0.5, 0.5, 0.25, 0.5, 0.0])
        #agent.model.baseline = dict(state=Baseline())
        #agent.model.baseline['state'].predict = lambda states: baseline

        #result, _ = agent.model.reward_estimation(states=dict(state=states), rewards=rewards, terminals=terminals)
        expected = discounted_rewards - baseline
        #print(result)
        #print(expected)
        #self.assertTrue((result == expected).all())

    def test_gae(self):
        kwargs = dict(
            discount=0.75,
            batch_size=8,
            learning_rate=0.001,
            # gae_rewards=True,
            # gae_lambda=0.5
        )
        # agent = VPGAgent(
        #     states_spec=dict(shape=(1,)),
        #     actions_spec=dict(type='int', num_actions=2),
        #     network_spec=[dict(type='dense', size=32)],
        #     config=config
        # )

        states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rewards = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        terminals = [False, False, False, False, True, False, False, False, True]
        baseline = np.array([0.25, 0.5, 0.0, 0.25, 0.5, 0.5, 0.25, 0.5, 0.0])
        #agent.model.baseline = dict(state=Baseline())
        #agent.model.baseline['state'].predict = lambda states: baseline
        td_residuals = np.array([
            0.75 * 0.5 - 0.25, 1.0 - 0.5, 0.75 * 0.25, 0.75 * 0.5 - 0.25, 1.0,
            1.0 + 0.75 * 0.25 - 0.5, 0.75 * 0.5 - 0.25, 1.0 - 0.5, 0.0
        ])

        # result, _ = agent.model.reward_estimation(
        #     states=dict(state=states),
        #     rewards=rewards,
        #     terminals=terminals
        # )

        expected = np.array([
            np.sum(((0.5 * 0.75) ** np.array([0, 1, 2, 3, 4])) * td_residuals[:5]),
            np.sum(((0.5 * 0.75) ** np.array([0, 1, 2, 3])) * td_residuals[1:5]),
            np.sum(((0.5 * 0.75) ** np.array([0, 1, 2])) * td_residuals[2:5]),
            np.sum(((0.5 * 0.75) ** np.array([0, 1])) * td_residuals[3:5]),
            np.sum(((0.5 * 0.75) ** np.array([0])) * td_residuals[4:5]),
            np.sum(((0.5 * 0.75) ** np.array([0, 1, 2, 3])) * td_residuals[5:]),
            np.sum(((0.5 * 0.75) ** np.array([0, 1, 2])) * td_residuals[6:]),
            np.sum(((0.5 * 0.75) ** np.array([0, 1])) * td_residuals[7:]),
            np.sum(((0.5 * 0.75) ** np.array([0])) * td_residuals[8:])
        ])
        #self.assertTrue((result == expected).all())
