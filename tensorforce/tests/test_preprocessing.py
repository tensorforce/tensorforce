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

"""
Preprocessor testing.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange
import unittest

import numpy as np

from tensorforce import preprocessing


class TestPreprocessing(unittest.TestCase):

    def test_preprocessing_grayscale(self):
        """
        Testing grayscale preprocessor. Verifies expected and calculated state shapes.
        """
        pp = preprocessing.grayscale.Grayscale()

        shape = list(np.random.randint(1, 8, size=2)) + [3]
        state = np.random.randint(0, 255, size=shape)

        # verify expected shape
        processed_shape = pp.shape(shape)
        self.assertEqual(tuple(processed_shape), tuple(shape[0:2]))

        # verify calculated shape
        processed_state = pp.process(state)
        self.assertEqual(processed_state.shape, tuple(processed_shape))

    def test_preprocessing_concat(self):
        """
        Testing concat preprocessor. Verifies expected and calculated state shapes.
        """
        concat_length = np.random.randint(1, 10)

        pp = preprocessing.concat.Concat(concat_length)

        shape = list(np.random.randint(1, 8, size=3))
        state = np.random.randint(0, 255, size=shape)

        # verify expected shape
        processed_shape = pp.shape(shape)
        self.assertEqual(tuple(processed_shape), tuple([concat_length] + shape))

        # verify calculated shape
        processed_state = pp.process(state)
        self.assertEqual(processed_state.shape, tuple(processed_shape))

        # verify calculated content
        states = [state]
        for i in xrange(concat_length-1):
            new_state = np.random.randint(0, 255, size=shape)
            states.append(new_state)
            processed_state = pp.process(new_state)

        self.assertFalse((np.array(states) - processed_state).any())

        # add another state
        new_state = np.random.randint(0, 255, size=shape)
        states.append(new_state)
        processed_state = pp.process(new_state)

        self.assertFalse((np.array(states[1:]) - processed_state).any())

    def test_preprocessing_imresize(self):
        """
        Testing imresize preprocessor. Verifies expected and calculated state shapes.
        """
        dimensions = list(np.random.randint(4, 8, size=2))

        pp = preprocessing.imresize.Imresize(*dimensions)

        shape = list(np.random.randint(1, 8, size=2))
        state = np.random.randint(0, 255, size=shape)

        # verify expected shape
        processed_shape = pp.shape(shape)
        self.assertEqual(tuple(processed_shape), tuple(dimensions))

        # verify calculated shape
        processed_state = pp.process(state)
        self.assertEqual(processed_state.shape, tuple(processed_shape))

    def test_preprocessing_maximum(self):
        """
        Testing maximum preprocessor. Verifies expected and calculated state shapes.
        """
        count = np.random.randint(1, 10)

        pp = preprocessing.maximum.Maximum(count)

        shape = list(np.random.randint(1, 8, size=3))
        state = np.random.randint(0, 255, size=shape)

        # verify expected shape
        processed_shape = pp.shape(shape)
        self.assertEqual(tuple(processed_shape), tuple(shape))

        # verify calculated shape
        processed_state = pp.process(state)
        self.assertEqual(processed_state.shape, tuple(processed_shape))

        # verify calculated content
        states = [state]
        max_state = state.reshape(-1)
        for i in xrange(count - 1):
            new_state = np.random.randint(0, 255, size=shape)

            new_state_reshaped = new_state.reshape(-1)

            # find maximum values manually
            for j, val in enumerate(new_state_reshaped):
                if max_state[j] < val:
                    max_state[j] = val

            states.append(new_state)
            processed_state = pp.process(new_state)

        self.assertFalse((max_state.reshape(shape) - processed_state).any())
