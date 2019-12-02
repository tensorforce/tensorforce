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

import unittest

from tensorforce.core.memories import Replay
from tensorforce.core.networks import LayerbasedNetwork
from test.unittest_base import UnittestBase


class TestNetwork(LayerbasedNetwork):

    def __init__(self, name, inputs_spec):
        super().__init__(name=name, inputs_spec=inputs_spec)

        self.layer1 = self.add_module(name='dense0', module=dict(type='dense', size=8))
        self.layer2 = self.add_module(name='dense1', module=dict(type='dense', size=8))

    def tf_apply(self, x, internals, return_internals=False):
        x = self.layer2.apply(x=self.layer1.apply(x=next(iter(x.values()))))
        if return_internals:
            return x, dict()
        else:
            return x


class TestSpecifications(UnittestBase, unittest.TestCase):

    def specification_unittest(self, network, memory):
        states = dict(type='float', shape=(3,))

        agent, environment = self.prepare(
            states=states, policy=dict(network=network), memory=memory
        )

        states = environment.reset()
        actions = agent.act(states=states, independent=True)
        states, terminal, reward = environment.execute(actions=actions)
        # agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        self.finished_test()

    def test_specifications(self):
        # SPECIFICATION.MD
        self.start_tests()

        # default
        self.specification_unittest(
            network=dict(type='layered', layers=[dict(type='dense', size=8)]),
            memory=dict(type='replay', capacity=100)
        )

        # json
        self.specification_unittest(
            network='test/data/network.json',
            memory=dict(type='test/data/memory.json', capacity=100)
        )

        # module
        self.specification_unittest(
            network='test.test_specifications.TestNetwork',
            memory=dict(type='tensorforce.core.memories.Replay', capacity=100)
        )

        # callable
        self.specification_unittest(
            network=TestNetwork, memory=dict(type=Replay, capacity=100)
        )

        # default (+firstarg)
        self.specification_unittest(
            network=[dict(type='dense', size=8)], memory=dict(capacity=100)
        )
