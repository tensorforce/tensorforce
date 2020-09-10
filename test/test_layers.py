# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

    def test_convolution(self):
        self.start_tests(name='convolution')

        states = dict(type='float', shape=(2, 3), min_value=1.0, max_value=2.0)
        actions = dict(
            bool_action=dict(type='bool', shape=(2,)),
            int_action=dict(type='int', shape=(2, 2), num_values=4),
            float_action=dict(type='float', shape=(2,), min_value=1.0, max_value=2.0),
            beta_action=dict(type='float', shape=(2, 2), min_value=1.0, max_value=2.0)
        )
        network = [
            dict(type='conv1d', size=8),
            dict(type='conv1d_transpose', size=8),
            dict(type='linear', size=8)
        ]
        self.unittest(states=states, actions=actions, policy=network)

        states = dict(type='float', shape=(2, 2, 3), min_value=1.0, max_value=2.0)
        actions = dict(
            bool_action=dict(type='bool', shape=(2, 2)),
            int_action=dict(type='int', shape=(2, 2, 2), num_values=4),
            float_action=dict(type='float', shape=(2, 2), min_value=1.0, max_value=2.0),
            beta_action=dict(type='float', shape=(2, 2, 2), min_value=1.0, max_value=2.0)
        )
        network = [
            dict(type='conv2d', size=8),
            dict(type='conv2d_transpose', size=8),
            dict(type='linear', size=8)
        ]
        self.unittest(states=states, actions=actions, policy=network)

    def test_dense(self):
        self.start_tests(name='dense')

        states = dict(type='float', shape=(3,), min_value=1.0, max_value=2.0)
        network = [
            dict(type='dense', size=8),
            dict(type='linear', size=8)
        ]
        self.unittest(states=states, policy=network)

    def test_embedding(self):
        self.start_tests(name='embedding')

        states = dict(type='int', shape=(), num_values=5)
        network = [dict(type='embedding', size=8)]
        self.unittest(states=states, policy=network)

    def test_input_rnn(self):
        self.start_tests(name='input-rnn')

        states = dict(type='float', shape=(2, 3), min_value=1.0, max_value=2.0)
        network = [
            dict(type='input_rnn', cell='gru', size=8, return_final_state=False),
            dict(type='input_lstm', size=8)
        ]
        self.unittest(states=states, policy=network)

    def test_keras(self):
        self.start_tests(name='keras')

        states = dict(type='float', shape=(3,), min_value=1.0, max_value=2.0)
        network = [dict(type='keras', layer='Dense', units=8)]
        self.unittest(states=states, policy=network)

    def test_misc(self):
        self.start_tests(name='misc')

        states = dict(type='float', shape=(3, 2), min_value=1.0, max_value=2.0)
        network = [
            dict(type='activation', nonlinearity='tanh'),
            dict(type='dropout', rate=0.5),
            (lambda x: x + 1.0),
            dict(type='reshape', shape=6),
            dict(
                type='function', function=(lambda x: x[:, :2]),
                output_spec=dict(type='float', shape=(2,))
            )
        ]
        self.unittest(states=states, policy=network)

        states = dict(type='float', shape=(3,), min_value=1.0, max_value=2.0)
        network = [
            dict(type='block', name='test', layers=[
                dict(type='dense', size=4), dict(type='dense', size=3)
            ]),
            dict(type='reuse', layer='test')
        ]
        self.unittest(states=states, policy=network)

        states = dict(type='float', shape=(3,), min_value=1.0, max_value=2.0)
        network = [
            dict(type='register', tensor='test'),
            dict(type='retrieve', tensors=('test',)),
            dict(type='retrieve', tensors=('state', 'test'), aggregation='product')
        ]
        self.unittest(states=states, policy=network)

    def test_normalization(self):
        self.start_tests(name='normalization')

        states = dict(type='float', shape=(3,), min_value=1.0, max_value=2.0)
        network = [
            dict(type='linear_normalization'),
            dict(type='exponential_normalization', decay=0.99),
            dict(type='instance_normalization')
        ]
        # 'batch_normalization' used by all tests
        self.unittest(states=states, policy=network)

    def test_pooling(self):
        self.start_tests(name='pooling')

        states = dict(type='float', shape=(2, 3), min_value=1.0, max_value=2.0)
        network = [
            dict(type='pool1d', reduction='average'),
            dict(type='flatten')
        ]
        self.unittest(states=states, policy=network)

        states = dict(type='float', shape=(2, 2, 3), min_value=1.0, max_value=2.0)
        network = [
            dict(type='pool2d', reduction='max'),
            dict(type='pooling', reduction='max')
        ]
        self.unittest(states=states, policy=network)

    def test_preprocessing(self):
        self.start_tests(name='preprocessing')

        states = dict(type='float', shape=(), min_value=-1.0, max_value=2.0)
        state_preprocessing = [
            dict(type='sequence', length=3, concatenate=False),
            dict(type='clipping', lower=-1.0, upper=1.0),
            dict(type='linear_normalization')
        ]
        reward_preprocessing = [dict(type='clipping', upper=1.0)]
        network = [dict(type='dense', size=8)]
        self.unittest(
            states=states, experience_update=False, policy=network,
            state_preprocessing=state_preprocessing, reward_preprocessing=reward_preprocessing
        )

        states = dict(state=dict(type='float', shape=(4, 4, 3), min_value=1.0, max_value=2.0))
        state_preprocessing = dict(state=[
            dict(type='image', height=2, width=2, grayscale=True),
            dict(type='deltafier', concatenate=0),
            dict(type='sequence', length=4),
            dict(type='linear_normalization')
        ])
        reward_preprocessing = dict(type='deltafier')
        network = [dict(type='reshape', shape=32)]
        # TODO: buffer_observe incompatible with Deltafier/Sequence expecting single-step inputs
        self.unittest(
            states=states, experience_update=False, policy=network,
            state_preprocessing=state_preprocessing, reward_preprocessing=reward_preprocessing,
            config=dict(
                buffer_observe=1, eager_mode=True, create_debug_assertions=True, tf_log_level=20
            )
        )

    def test_rnn(self):
        self.start_tests(name='rnn')

        states = dict(type='float', shape=(3,), min_value=1.0, max_value=2.0)
        network = [
            dict(type='rnn', cell='gru', size=8, horizon=2),
            dict(type='lstm', size=7, horizon=1)
        ]
        self.unittest(states=states, policy=network)
