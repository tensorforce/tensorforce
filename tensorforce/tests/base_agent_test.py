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

from tensorforce.tests.base_test import BaseTest
from tensorforce.core.networks import Dense, LayerBasedNetwork
from tensorforce.tests.minimal_test import MinimalTest


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

        environment = MinimalTest(specification={'bool': ()})

        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        self.base_test_pass(
            name='bool',
            environment=environment,
            network=network,
            **self.__class__.config
        )

    def test_int(self):
        """
        Tests the case of one integer action.
        """
        if self.__class__.exclude_int:
            return

        environment = MinimalTest(specification={'int': ()})
        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        self.base_test_pass(
            name='int',
            environment=environment,
            network=network,
            **self.__class__.config
        )

    def test_float(self):
        """
        Tests the case of one float action.
        """
        if self.__class__.exclude_float:
            return

        environment = MinimalTest(specification={'float': ()})
        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        self.base_test_pass(
            name='float',
            environment=environment,
            network=network,
            **self.__class__.config
        )

    def test_bounded_float(self):
        """
        Tests the case of one bounded float action, i.e. with min and max value.
        """
        if self.__class__.exclude_bounded:
            return

        environment = MinimalTest(specification={'bounded': ()})
        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        self.base_test_pass(
            name='bounded',
            environment=environment,
            network=network,
            **self.__class__.config
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

        network = list()
        if not exclude_bool:
            network.append([
                dict(type='input', names='bool'),
                dict(type='dense', size=16),
                dict(type='dense', size=16),
                dict(type='output', name='bool-emb')
            ])
        if not exclude_int:
            network.append([
                dict(type='input', names='int'),
                dict(type='dense', size=16),
                dict(type='dense', size=16),
                dict(type='output', name='int-emb')
            ])
        if not exclude_float:
            network.append([
                dict(type='input', names='float'),
                dict(type='dense', size=16),
                dict(type='dense', size=16),
                dict(type='output', name='float-emb')
            ])
        if not exclude_bounded:
            network.append([
                dict(type='input', names='bounded'),
                dict(type='dense', size=16),
                dict(type='dense', size=16),
                dict(type='output', name='bounded-emb')
            ])
        network.append([
            dict(type='input', names=['bool-emb', 'int-emb', 'float-emb', 'bounded-emb'], aggregation_type='product'),
            dict(type='dense', size=16)
        ])

        # class CustomNetwork(LayerBasedNetwork):

        #     def __init__(self, scope='layerbased-network', summary_labels=()):
        #         super(CustomNetwork, self).__init__(scope=scope, summary_labels=summary_labels)

        #         if not exclude_bool:
        #             self.layer_bool1 = Dense(size=16, scope='state-bool1')
        #             self.add_layer(layer=self.layer_bool1)
        #             self.layer_bool2 = Dense(size=16, scope='state-bool2')
        #             self.add_layer(layer=self.layer_bool2)

        #         if not exclude_int:
        #             self.layer_int1 = Dense(size=16, scope='state-int1')
        #             self.add_layer(layer=self.layer_int1)
        #             self.layer_int2 = Dense(size=16, scope='state-int2')
        #             self.add_layer(layer=self.layer_int2)

        #         if not exclude_float:
        #             self.layer_float1 = Dense(size=16, scope='state-float1')
        #             self.add_layer(layer=self.layer_float1)
        #             self.layer_float2 = Dense(size=16, scope='state-float2')
        #             self.add_layer(layer=self.layer_float2)

        #         if not exclude_bounded:
        #             self.layer_bounded1 = Dense(size=16, scope='state-bounded1')
        #             self.add_layer(layer=self.layer_bounded1)
        #             self.layer_bounded2 = Dense(size=16, scope='state-bounded2')
        #             self.add_layer(layer=self.layer_bounded2)

        #     def tf_apply(self, x, internals, update, return_internals=False):
        #         xs = list()

        #         if not exclude_bool:
        #             xs.append(self.layer_bool2.apply(
        #                 x=self.layer_bool1.apply(x=x['bool'], update=update), update=update
        #             ))

        #         if not exclude_int:
        #             xs.append(self.layer_int2.apply(
        #                 x=self.layer_int1.apply(x=x['int'], update=update), update=update
        #             ))

        #         if not exclude_float:
        #             xs.append(self.layer_float2.apply(
        #                 x=self.layer_float1.apply(x=x['float'], update=update), update=update
        #             ))

        #         if not exclude_bounded:
        #             xs.append(self.layer_bounded2.apply(
        #                 x=self.layer_bounded1.apply(x=x['bounded'], update=update), update=update
        #             ))

        #         x = xs[0]
        #         for y in xs[1:]:
        #             x *= y

        #         if return_internals:
        #             return x, dict()
        #         else:
        #             return x

        specification = dict()
        if not exclude_bool:
            specification['bool'] = ()
        if not exclude_int:
            specification['int'] = (2,)
        if not exclude_float:
            specification['float'] = (1, 1)
        if not exclude_bounded:
            specification['bounded'] = (1,)

        environment = MinimalTest(specification=specification)

        self.base_test_run(
            name='multi',
            environment=environment,
            network=network,
            # network=CustomNetwork,
            **self.__class__.config
        )

    def test_lstm(self):
        """
        Tests the case of using internal states via an LSTM layer (for one integer action).
        """
        if self.__class__.exclude_lstm:
            return

        environment = MinimalTest(specification={'int': ()})
        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32),
            dict(type='internal_lstm', size=32)
        ]

        self.base_test_pass(
            name='lstm',
            environment=environment,
            network=network,
            **self.__class__.config
        )
