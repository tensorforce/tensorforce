# Copyright 2020 Tensorforce Team. All Rights Reserved.
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
from tensorforce.core import tf_util
from test.unittest_base import UnittestBase


class TestPrecision(UnittestBase, unittest.TestCase):

    def test_precision(self):
        self.start_tests()

        try:
            util.np_dtype_mapping = dict(bool=np.bool_, int=np.int32, float=np.float16)
            tf_util.DTYPE_MAPPING = dict(bool=tf.bool, int=tf.int32, float=tf.float16)

            # TODO: TensorFlow optimizers seem incompatible with float16
            optimizer = dict(optimizer='evolutionary', learning_rate=1e-3)
            baseline_optimizer = dict(optimizer='evolutionary', learning_rate=1e-3)
            self.unittest(
                optimizer=optimizer, baseline_optimizer=baseline_optimizer,
                config=dict(eager_mode=True, create_debug_assertions=True, tf_log_level=20)
            )

            util.np_dtype_mapping = dict(bool=np.bool_, int=np.int64, float=np.float64)
            tf_util.DTYPE_MAPPING = dict(bool=tf.bool, int=tf.int64, float=tf.float64)

            self.unittest()

        except BaseException as exc:
            raise exc
            self.assertTrue(expr=False)

        finally:
            util.np_dtype_mapping = dict(bool=np.bool_, int=np.int64, float=np.float32)
            tf_util.DTYPE_MAPPING = dict(bool=tf.bool, int=tf.int64, float=tf.float32)
