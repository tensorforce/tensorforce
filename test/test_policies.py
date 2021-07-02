# Copyright 2021 Tensorforce Team. All Rights Reserved.
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


class TestPolicies(UnittestBase, unittest.TestCase):

    def test_keras_network(self):
        self.start_tests(name='keras network')

        import tensorflow as tf

        class Model(tf.keras.Model):

            def __init__(self):
                super().__init__()
                self.layer1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
                self.layer2 = tf.keras.layers.Dense(5, activation=tf.nn.relu)

            def call(self, inputs):
                x = self.layer1(inputs)
                return self.layer2(x)

        states = dict(type='float', shape=(3,), min_value=1.0, max_value=2.0)
        self.unittest(states=states, policy=Model)

        class Model(tf.keras.Model):

            def __init__(self, size):
                super().__init__()
                self.layer1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
                self.layer2 = tf.keras.layers.Embedding(4, 4)
                self.layer3 = tf.keras.layers.Dense(size, activation=tf.nn.relu)

            def call(self, inputs):
                y = self.layer1(inputs[0])
                x = self.layer2(inputs[1])
                return self.layer3(tf.concat(values=[x, y], axis=1))

        states = dict(
            int_state=dict(type='int', shape=(), num_values=4),
            float_state=dict(type='float', shape=(3,), min_value=1.0, max_value=2.0)
        )
        self.unittest(states=states, policy=dict(network=dict(type='keras', model=Model, size=5)))

        self.unittest(states=states, policy=Model(size=5))

        def model(size):
            return Model(size=size)

        self.unittest(states=states, policy=dict(network=dict(type='keras', model=model, size=5)))

    def test_multi_output(self):
        self.start_tests(name='multi-output')
        self.unittest(
            states=dict(
                state1=dict(type='float', shape=(2,), min_value=-1.0, max_value=2.0),
                state2=dict(type='float', shape=(3,), min_value=1.0, max_value=2.0)
            ),
            actions=dict(
                action1=dict(type='int', shape=(), num_values=3),
                action2=dict(type='int', shape=(), num_values=4)
            ),
            policy=dict(network=[
                [
                    dict(type='retrieve', tensors=['state1']),
                    dict(type='dense', size=16),
                    dict(type='register', tensor='action1-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['state2', 'action1-embedding']),
                    dict(type='dense', size=12)
                ]
            ], single_output=False),
            baseline=dict(type='parametrized_value_policy', network=[
                [
                    dict(type='retrieve', tensors=['state1']),
                    dict(type='dense', size=16),
                    dict(type='register', tensor='action1-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['state2', 'action1-embedding']),
                    dict(type='dense', size=12),
                    dict(type='register', tensor='state-embedding')
                ]
            ], single_output=False),
        )

    def test_categorical_skip_linear(self):
        self.start_tests(name='categorical skip-linear')
        self.unittest(
            states=dict(type='float', shape=(3,), min_value=1.0, max_value=2.0),
            actions=dict(type='int', shape=(2,), num_values=4),
            policy=dict(
                network=[dict(type='dense', size=8), dict(type='reshape', shape=(2, 4))],
                distributions=dict(type='categorical', skip_linear=True)
            )
        )

    def test_categorical_skip_linear_no_shape(self):
        self.start_tests(name='categorical skip-linear empty shape')
        self.unittest(
            states=dict(type='float', shape=(3,), min_value=1.0, max_value=2.0),
            actions=dict(type='int', num_values=4),
            policy=dict(
                network=[dict(type='dense', size=4)],
                distributions=dict(type='categorical', skip_linear=True)
            )
        )
