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
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
from six.moves import xrange
import tensorforce.core.preprocessing


class TestPreprocessing(unittest.TestCase):

    def _test_preprocessing_shape(self, pp, shape, expected_shape, state=None):
        if state is None:
            state = np.random.randint(0, 255, size=shape)

        # verify expected shape
        processed_shape = pp.shape(shape)
        self.assertEqual(tuple(processed_shape), expected_shape)

        # verify calculated shape
        processed_state = pp.process(state)
        self.assertEqual(processed_state.shape, tuple(processed_shape))

        return processed_state, processed_shape

    def test_preprocessing_grayscale(self):
        """
        Testing grayscale preprocessor. Verifies expected and calculated state shapes.
        """
        pp = tensorforce.core.preprocessing.grayscale.Grayscale()

        shape = list(np.random.randint(1, 8, size=2)) + [3]

        processed_state, processed_shape = self._test_preprocessing_shape(pp, shape, tuple(shape[0:2]))

    def test_preprocessing_concat(self):
        """
        Testing concat preprocessor. Verifies expected and calculated state shapes.
        """
        concat_length = np.random.randint(1, 10)

        pp = tensorforce.core.preprocessing.concat.Concat(concat_length)

        shape = list(np.random.randint(1, 8, size=3))
        state = np.random.randint(0, 255, size=shape)

        processed_state, processed_shape = self._test_preprocessing_shape(pp, shape, tuple([concat_length] + shape), state)

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

        pp = tensorforce.core.preprocessing.imresize.Imresize(*dimensions)

        shape = list(np.random.randint(1, 8, size=2))

        processed_state, processed_shape = self._test_preprocessing_shape(pp, shape, tuple(dimensions))

    def test_preprocessing_maximum(self):
        """
        Testing maximum preprocessor. Verifies expected and calculated state shapes.
        """
        count = np.random.randint(1, 10)

        pp = tensorforce.core.preprocessing.maximum.Maximum(count)

        shape = list(np.random.randint(1, 8, size=3))
        state = np.random.randint(0, 255, size=shape)

        processed_state, processed_shape = self._test_preprocessing_shape(pp, shape, tuple(shape), state)

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

    def test_preprocessing_multistack(self):
        concat_length = np.random.randint(1, 10)

        pp_config = {
            'state1': [
                ['maximum', 2]
            ],
            'state2': [
                ['concat', concat_length],
                ['normalize']
            ]
        }

        shape = list(np.random.randint(1, 8, size=2)) + [3]

        expected = dict(
            state1=tuple(shape),
            state2=tuple([concat_length] + shape)
        )

        stack = tensorforce.core.preprocessing.build_preprocessing_stack(pp_config)

        self.assertTrue(isinstance(stack, tensorforce.core.preprocessing.MultiStack))

        input_states = dict()
        for state_name, state_config in pp_config.items():
            state = np.random.randint(0, 255, size=shape)

            input_states.update({state_name: state})

        processed_states = stack.process(input_states)

        for state_name, state in processed_states.items():
            self.assertEqual(expected[state_name], state.shape)


