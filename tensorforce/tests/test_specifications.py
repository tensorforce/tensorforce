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
import sys

from tensorforce.agents import VPGAgent
from tensorforce.core.networks import LayerbasedNetwork, LayeredNetwork
from tensorforce.tests.agent_unittest import UnittestBase


class TestNetwork(LayerbasedNetwork):

    def __init__(self, name, inputs_spec):
        super().__init__(name=name, inputs_spec=inputs_spec)

        self.layer1 = self.add_module(name='dense0', module=dict(type='dense', size=32))
        self.layer2 = self.add_module(name='dense1', module=dict(type='dense', size=32))

    def tf_apply(self, x, internals, return_internals=False):
        x = self.layer2.apply(x=self.layer1.apply(x=next(iter(x.values()))))
        if return_internals:
            return x, dict()
        else:
            return x


class TestSpecifications(UnittestBase, unittest.TestCase):

    agent = VPGAgent

    def specification_unittest(self, name, network):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        agent, environment = self.prepare(
            name=name, states=states, actions=actions, network=network
        )

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()
        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_spec_default(self):
        spec_without_type = dict(
            layers=[dict(type='dense', size=32), dict(type='dense', size=32)]
        )
        self.specification_unittest(name='spec-default', network=spec_without_type)

    def test_spec_callable(self):
        spec_with_callable = dict(
            type=LayeredNetwork, layers=[dict(type='dense', size=32), dict(type='dense', size=32)]
        )
        self.specification_unittest(name='spec-callable', network=spec_with_callable)

    def test_spec_json(self):
        spec_with_json = dict(
            type='tensorforce/tests/network.json',
            layers=[dict(type='dense', size=32), dict(type='dense', size=32)]
        )
        self.specification_unittest(name='spec-json', network=spec_with_json)

    def test_spec_module(self):
        spec_with_module = dict(
            type='tensorforce.core.networks.layered.LayeredNetwork',
            layers=[dict(type='dense', size=32), dict(type='dense', size=32)]
        )
        self.specification_unittest(name='spec-module', network=spec_with_module)

    def test_callable(self):
        self.specification_unittest(name='callable', network=TestNetwork)

    def test_json(self):
        json_file = 'tensorforce/tests/network.json'
        self.specification_unittest(name='json', network=json_file)

    def test_module(self):
        module = 'tensorforce.tests.test_specifications.TestNetwork'
        self.specification_unittest(name='module', network=module)

    def test_firstarg(self):
        layers = [dict(type='dense', size=32), dict(type='dense', size=32)]
        self.specification_unittest(name='firstarg', network=layers)
