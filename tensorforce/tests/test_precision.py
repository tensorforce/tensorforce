# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest

import numpy as np
import tensorflow as tf

from tensorforce import util
from tensorforce.agents import DQNAgent, VPGAgent
from tensorforce.tests.unittest_base import UnittestBase


class TestPrecision(UnittestBase, unittest.TestCase):

    config = dict(update_mode=dict(batch_size=2))

    def test_precision_dqn(self):
        try:
            util.np_dtype_mapping = dict(
                bool=np.bool_, int=np.int16, long=np.int32, float=np.float16
            )
            util.tf_dtype_mapping = dict(
                bool=tf.bool, int=tf.int16, long=tf.int32, float=tf.float16
            )

            states = dict(type='float', shape=(1,))

            actions = dict(type='int', shape=(), num_values=3)

            network = dict(type='auto', internal_rnn=False)

            self.unittest(
                name='precision-dqn', states=states, actions=actions, agent=DQNAgent,
                network=network
            )

        except Exception:
            self.assertTrue(expr=False)

        finally:
            util.np_dtype_mapping = dict(
                bool=np.bool_, int=np.int32, long=np.int64, float=np.float32
            )
            util.tf_dtype_mapping = dict(
                bool=tf.bool, int=tf.int32, long=tf.int64, float=tf.float32
            )

    def test_precision_vpg(self):
        try:
            util.np_dtype_mapping = dict(
                bool=np.bool_, int=np.int16, long=np.int32, float=np.float16
            )
            util.tf_dtype_mapping = dict(
                bool=tf.bool, int=tf.int16, long=tf.int32, float=tf.float16
            )

            states = dict(type='int', shape=(), num_values=3)

            actions = dict(type='float', shape=())

            network = dict(type='auto', internal_rnn=False)

            self.unittest(
                name='precision-vpg', states=states, actions=actions, agent=VPGAgent, network=network
            )

        except Exception:
            self.assertTrue(expr=False)

        finally:
            util.np_dtype_mapping = dict(
                bool=np.bool_, int=np.int32, long=np.int64, float=np.float32
            )
            util.tf_dtype_mapping = dict(
                bool=tf.bool, int=tf.int32, long=tf.int64, float=tf.float32
            )
