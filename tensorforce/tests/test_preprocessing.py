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
from __future__ import division
from __future__ import print_function

from random import randint
import unittest

import numpy as np

from tensorforce.core.preprocessing import Sequence, Standardize, Normalize, Grayscale, ImageResize, Divide


class TestPreprocessing(unittest.TestCase):

    def test_preprocessing_sequence(self):
        """
        Tests whether the sequence preprocessor concatenates individual
        states into the correct shape.

        """
        length = 3
        sequence = Sequence(length=length)
        shape = (randint(1, 1), randint(1, 1), 3)
        state1 = np.random.randint(0, 256, shape)
        state2 = np.random.randint(0, 256, shape)
        state3 = np.random.randint(0, 256, shape)
        state4 = np.random.randint(0, 256, shape)
        processed_state1 = sequence.process(state1)
        processed_state2 = sequence.process(state2)
        processed_state3 = sequence.process(state3)
        processed_state4 = sequence.process(state4)

        self.assertEqual(sequence.processed_shape(shape), (shape[0], shape[1], shape[2] * length))
        self.assertEqual(processed_state1.shape, sequence.processed_shape(shape))
        self.assertEqual(processed_state2.shape, sequence.processed_shape(shape))
        self.assertEqual(processed_state3.shape, sequence.processed_shape(shape))
        self.assertEqual(processed_state4.shape, sequence.processed_shape(shape))

        self.assertTrue(np.all(processed_state1 == np.concatenate((state1, state1, state1), -1)))
        self.assertTrue(np.all(processed_state2 == np.concatenate((state1, state1, state2), -1)))
        self.assertTrue(np.all(processed_state3 == np.concatenate((state1, state2, state3), -1)))
        self.assertTrue(np.all(processed_state4 == np.concatenate((state2, state3, state4), -1)))

    def test_preprocessing_normalize(self):
        normalize = Standardize()
        shape = (randint(1, 64), randint(1, 64), 3)
        state = np.random.randint(0, 256, shape)

        self.assertEqual(normalize.processed_shape(shape), shape)
        self.assertEqual(normalize.process(state).shape, normalize.processed_shape(shape))

    def test_preprocessing_divide(self):
        scale = 3
        divide = Divide(scale)
        shape = (randint(1, 64), randint(1, 64), 3)
        state = np.random.randint(0, 256, shape)

        self.assertEqual(divide.processed_shape(shape), shape)
        self.assertEqual(divide.process(state).shape, divide.processed_shape(shape))
        self.assertTrue(np.allclose(divide.process(state), state / scale))


    def test_preprocessing_center(self):
        center = Normalize()
        shape = (randint(1, 64), randint(1, 64), 3)
        state = np.random.randint(0, 256, shape)

        self.assertEqual(center.processed_shape(shape), shape)
        self.assertEqual(center.process(state).shape, center.processed_shape(shape))

    def test_preprocessing_grayscale(self):
        grayscale = Grayscale()
        shape = (randint(1, 64), randint(1, 64), 3)
        state = np.random.randint(0, 256, shape)

        self.assertEqual(grayscale.processed_shape(shape), (shape[0], shape[1], 1))
        self.assertEqual(grayscale.process(state).shape, grayscale.processed_shape(shape))

    def test_preprocessing_image_resize(self):
        width = randint(1, 64)
        height = randint(1, 64)
        image_resize = ImageResize(width=width, height=height)
        shape = (randint(1, 64), randint(1, 64), 3)
        state = np.random.randint(0, 256, shape)

        self.assertEqual(image_resize.processed_shape(shape), (width, height, 3))
        self.assertEqual(image_resize.process(state).shape, image_resize.processed_shape(shape))
