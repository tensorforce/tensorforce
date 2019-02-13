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

from tensorforce.agents import VPGAgent
from tensorforce.tests.unittest_base import UnittestBase


class TestLayers(UnittestBase, unittest.TestCase):

    agent = VPGAgent
    config = dict(update_mode=dict(batch_size=2))

    def test_dropout(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [
            dict(type='dense', size=32), dict(type='dropout', rate=0.5),
            dict(type='dense', size=32)
        ]

        self.unittest(name='dropout', states=states, actions=actions, network=network)

    def test_rnn(self):
        states = dict(
            bool_state=dict(type='bool', shape=(3,)),
            int_state=dict(type='int', shape=(2,), num_values=4)
        )

        actions = dict(type='int', shape=(), num_values=3)

        network = [
            [
                dict(type='retrieve', tensors='bool_state'),
                dict(type='embedding', size=16),
                dict(type='gru', size=16, return_final_state=False),
                dict(type='gru', size=16),
                dict(type='register', tensor='bool-emb')
            ],
            [
                dict(type='retrieve', tensors='int_state'),
                dict(type='embedding', size=16),
                dict(type='lstm', size=16, return_final_state=False),
                dict(type='lstm', size=16),
                dict(type='register', tensor='int-emb')
            ],
            [
                dict(type='retrieve', tensors=('bool-emb', 'int-emb'), aggregation='product'),
                dict(type='dense', size=16)
            ]
        ]

        self.unittest(name='rnn', states=states, actions=actions, network=network)

    def test_keras(self):
        states = dict(
            bool_state=dict(type='bool', shape=(3,)),
            int_state=dict(type='int', shape=(2,), num_values=4),
            float_state=dict(type='float', shape=(4, 4, 1)),
            bounded_state=dict(type='float', shape=(2,), min_value=-0.5, max_value=0.5)
        )

        actions = dict(type='int', shape=(), num_values=3)

        network = [
            [
                dict(type='retrieve', tensors='bool_state'),
                dict(type='keras', layer='Embedding', input_dim=2, output_dim=16),
                dict(type='keras', layer='Conv1D', filters=16, kernel_size=3),
                dict(type='keras', layer='GlobalMaxPool1D'),
                dict(type='register', tensor='bool-emb')
            ],
            [
                dict(type='retrieve', tensors='int_state'),
                dict(type='keras', layer='Embedding', input_dim=4, output_dim=16),
                dict(type='keras', layer='LSTM', units=16),
                dict(type='register', tensor='int-emb')
            ],
            [
                dict(type='retrieve', tensors='float_state'),
                dict(type='keras', layer='Conv2D', filters=16, kernel_size=3),
                dict(type='keras', layer='MaxPool2D'),
                dict(type='keras', layer='GlobalMaxPool2D'),
                dict(type='register', tensor='float-emb')
            ],
            [
                dict(type='retrieve', tensors='bounded_state'),
                dict(type='keras', layer='Dense', units=16),
                dict(type='register', tensor='bounded-emb')
            ],
            [
                dict(
                    type='retrieve', tensors=('bool-emb', 'int-emb', 'float-emb', 'bounded-emb'),
                    aggregation='product'
                ),
                dict(type='keras', layer='Dense', units=16)
            ]
        ]

        self.unittest(name='keras', states=states, actions=actions, network=network)
