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

from tensorforce.core.networks import Dense, LayerBasedNetwork
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.tests.base_test import BaseTest


class BaseAgentTest(BaseTest):
    """
    Base class for tests of fundamental Agent functionality, i.e. various action types
    and shapes and internal states.
    """

    config = None

    # Exclude flags to indicate whether a certain test is excluded for a model.
    exclude_bool = False
    exclude_int = False
    exclude_float = False
    exclude_bounded = False
    exclude_multi = False
    exclude_lstm = False

    def test_bool(self):
        """
        Tests the case of one boolean action.
        """
        if self.__class__.exclude_bool:
            return

        environment = MinimalTest(specification=[('bool', ())])

        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        self.base_test(
            name='bool',
            environment=environment,
            network_spec=network_spec,
            config=self.__class__.config
        )

    def test_int(self):
        """
        Tests the case of one integer action.
        """
        if self.__class__.exclude_int:
            return

        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        self.base_test(
            name='int',
            environment=environment,
            network_spec=network_spec,
            config=self.__class__.config
        )

    def test_float(self):
        """
        Tests the case of one float action.
        """
        if self.__class__.exclude_float:
            return

        environment = MinimalTest(specification=[('float', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        self.base_test(
            name='float',
            environment=environment,
            network_spec=network_spec,
            config=self.__class__.config
        )

    def test_bounded_float(self):
        """
        Tests the case of one bounded float action, i.e. with min and max value.
        """
        if self.__class__.exclude_bounded:
            return

        environment = MinimalTest(specification=[('bounded-float', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        self.base_test(
            name='bounded-float',
            environment=environment,
            network_spec=network_spec,
            config=self.__class__.config
        )

    def test_multi(self):
        """
        Tests the case of multiple actions of different type and shape.
        """
        if self.__class__.exclude_multi:
            return

        exclude_bool = self.__class__.exclude_bool
        exclude_int = self.__class__.exclude_int
        exclude_float = self.__class__.exclude_float
        exclude_bounded = self.__class__.exclude_bounded

        class CustomNetwork(LayerBasedNetwork):

            def tf_apply(self, x, internals, return_internals=False):
                if exclude_bool:
                    x0 = 1.0
                else:
                    layer01 = Dense(size=32, scope='state0-1')
                    self.add_layer(layer=layer01)
                    layer02 = Dense(size=32, scope='state0-2')
                    self.add_layer(layer=layer02)
                    x0 = layer02.apply(x=layer01.apply(x=x['state0']))

                if exclude_int:
                    x1 = 1.0
                else:
                    layer11 = Dense(size=32, scope='state1-1')
                    self.add_layer(layer=layer11)
                    layer12 = Dense(size=32, scope='state1-2')
                    self.add_layer(layer=layer12)
                    x1 = layer12.apply(x=layer11.apply(x=x['state1']))

                if exclude_float:
                    x2 = 1.0
                else:
                    layer21 = Dense(size=32, scope='state2-1')
                    self.add_layer(layer=layer21)
                    layer22 = Dense(size=32, scope='state2-2')
                    self.add_layer(layer=layer22)
                    x2 = layer22.apply(x=layer21.apply(x=x['state2']))

                if exclude_bounded:
                    x3 = 1.0
                else:
                    layer31 = Dense(size=32, scope='state3-1')
                    self.add_layer(layer=layer31)
                    layer32 = Dense(size=32, scope='state3-2')
                    self.add_layer(layer=layer32)
                    x3 = layer32.apply(x=layer31.apply(x=x['state3']))

                x = x0 * x1 * x2 * x3
                return (x, list()) if return_internals else x

        specification = list()
        if not exclude_bool:
            specification.append(('bool', ()))
        if not exclude_int:
            specification.append(('int', (2,)))
        if not exclude_float:
            specification.append(('float', (1,)))
        if not exclude_bounded:
            specification.append(('bounded-float', (1, 1)))

        environment = MinimalTest(specification=specification)

        self.base_test(
            name='multi',
            environment=environment,
            network_spec=CustomNetwork,
            config=self.__class__.config
        )

    def test_lstm(self):
        """
        Tests the case of using internal states via an LSTM layer (for one integer action).
        """
        if self.__class__.exclude_lstm:
            return

        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32),
            dict(type='lstm', size=32)
        ]

        self.base_test(
            name='lstm',
            environment=environment,
            network_spec=network_spec,
            config=self.__class__.config
        )
