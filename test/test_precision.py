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

import pytest
import unittest

import numpy as np
import tensorflow as tf

from tensorforce import util
from test.unittest_base import UnittestBase


class TestPrecision(UnittestBase, unittest.TestCase):

    require_observe = True

    def test_precision(self):
        self.start_tests()

        try:
            # TODO: long=int32 since some operations like tf.math.maximum expect >= int32
            util.np_dtype_mapping = dict(
                bool=np.bool_, int=np.int16, long=np.int32, float=np.float16
            )
            util.tf_dtype_mapping = dict(
                bool=tf.bool, int=tf.int16, long=tf.int32, float=tf.float16
            )

            # TODO: Keras RNNs use float32 which causes mismatch during optimization
            self.unittest(
                policy=dict(network=dict(type='auto', size=8, depth=1, internal_rnn=False))
            )

        except BaseException as exc:
            raise exc
            self.assertTrue(expr=False)

        finally:
            util.np_dtype_mapping = dict(
                bool=np.bool_, int=np.int32, long=np.int64, float=np.float32
            )
            util.tf_dtype_mapping = dict(
                bool=tf.bool, int=tf.int32, long=tf.int64, float=tf.float32
            )
