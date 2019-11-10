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

from test.unittest_base import UnittestBase


class TestLayers(UnittestBase, unittest.TestCase):

    num_timesteps = 2

    def test_convolution(self):
        self.start_tests(name='convolution')

        states = dict(type='float', shape=(2, 3))
        actions = dict(
            bool_action=dict(type='bool', shape=(2,)),
            int_action=dict(type='int', shape=(2, 2), num_values=4),
            float_action=dict(type='float', shape=(2,)),
            bounded_action=dict(type='float', shape=(2, 2), min_value=-0.5, max_value=0.5)
        )
        network = [
            dict(type='conv1d', size=8),
            dict(type='conv1d_transpose', size=8),
            dict(type='linear', size=8)
        ]
        self.unittest(states=states, actions=actions, policy=dict(network=network))

        states = dict(type='float', shape=(2, 2, 3))
        actions = dict(
            bool_action=dict(type='bool', shape=(2, 2,)),
            int_action=dict(type='int', shape=(2, 2, 2), num_values=4),
            float_action=dict(type='float', shape=(2, 2,)),
            bounded_action=dict(type='float', shape=(2, 2, 2), min_value=-0.5, max_value=0.5)
        )
        network = [
            dict(type='conv2d', size=8),
            dict(type='conv2d_transpose', size=8),
            dict(type='linear', size=8)
        ]
        self.unittest(states=states, actions=actions, policy=dict(network=network))

    def test_dense(self):
        self.start_tests(name='dense')

        states = dict(type='float', shape=(3,))
        network = [
            dict(type='dense', size=8),
            dict(type='linear', size=8)
        ]
        self.unittest(states=states, policy=dict(network=network))

    def test_embedding(self):
        self.start_tests(name='embedding')

        states = dict(type='int', shape=(), num_values=5)
        network = [dict(type='embedding', size=8)]
        self.unittest(states=states, policy=dict(network=network))

    def test_internal_rnn(self):
        self.start_tests(name='internal-rnn')

        states = dict(type='float', shape=(3,))
        network = [
            dict(type='internal_rnn', cell='gru', size=8, length=2),
            dict(type='internal_lstm', size=8, length=2)
        ]
        self.unittest(states=states, policy=dict(network=network))

    def test_keras(self):
        self.start_tests(name='keras')

        states = dict(type='float', shape=(3,))
        network = [dict(type='keras', layer='Dense', units=8)]
        self.unittest(states=states, policy=dict(network=network))

    def test_misc(self):
        self.start_tests(name='misc')

        states = dict(type='float', shape=(3,))
        network = [
            dict(type='activation', nonlinearity='tanh'),
            dict(type='dropout', rate=0.5),
            dict(type='function', function=(lambda x: x + 1.0)),
            dict(function=(lambda x: x[:, :2]), output_spec=dict(shape=(2,)))
        ]
        self.unittest(states=states, policy=dict(network=network))

        states = dict(type='float', shape=(3,))
        network = [
            dict(type='block', name='test', layers=[
                dict(type='dense', size=3), dict(type='dense', size=3)
            ]),
            dict(type='reuse', layer='test')
        ]
        self.unittest(states=states, policy=dict(network=network))

        states = dict(type='float', shape=(3,))
        network = [
            dict(type='register', tensor='test'),
            dict(type='retrieve', tensors='test'),
            dict(type='retrieve', tensors=('*', 'test'), aggregation='product')
        ]
        self.unittest(states=states, policy=dict(network=network))

    def test_normalization(self):
        self.start_tests(name='normalization')

        states = dict(type='float', shape=(3,))
        network = [
            dict(type='exponential_normalization'),
            dict(type='instance_normalization')
        ]
        self.unittest(states=states, policy=dict(network=network))

    def test_pooling(self):
        self.start_tests(name='pooling')

        states = dict(type='float', shape=(2, 3))
        network = [
            dict(type='pool1d', reduction='average'),
            dict(type='flatten')
        ]
        self.unittest(states=states, policy=dict(network=network))

        states = dict(type='float', shape=(2, 2, 3))
        network = [
            dict(type='pool2d', reduction='max'),
            dict(type='pooling', reduction='max')
        ]
        self.unittest(states=states, policy=dict(network=network))

    def test_preprocessing(self):
        self.start_tests(name='preprocessing')

        states = dict(type='float', shape=(3,))
        preprocessing = dict(
            state=dict(type='clipping', lower=-1.0, upper=1.0),
            reward=[dict(type='clipping', lower=-1.0, upper=1.0)]
        )
        network = [dict(type='dense', name='test', size=8)]
        self.unittest(states=states, preprocessing=preprocessing, policy=dict(network=network))

        states = dict(type='float', shape=(4, 4, 3))
        preprocessing = dict(
            state=[
                dict(type='image', height=2, width=2, grayscale=True),
                dict(type='deltafier', concatenate=0)
            ],
            reward=dict(type='deltafier')
        )
        network = [dict(type='flatten')]
        self.unittest(states=states, preprocessing=preprocessing, policy=dict(network=network))

        # TODO: Sequence missing

    def test_rnn(self):
        self.start_tests(name='rnn')

        states = dict(type='float', shape=(2, 3))
        network = [
            dict(type='rnn', cell='gru', size=8, return_final_state=False),
            dict(type='lstm', size=8)
        ]
        self.unittest(states=states, policy=dict(network=network))
