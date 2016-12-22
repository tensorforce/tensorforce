# Copyright 2016 reinforce.io. All Rights Reserved.
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
Comment
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorforce import preprocessing

def test_preprocessing_grayscale():
    """
    Testing grayscale preprocessor. Verifies expected and calculated state shapes.
    """
    pp = preprocessing.grayscale.Grayscale([0.5, 0.2, 0.8])

    shape = list(np.random.randint(0, 200, size=2)) + [3]
    state = np.random.randint(0, 255, size=shape)

    processed_shape = pp.shape(shape)

    assert processed_shape == shape[0:2]

    processed_state = pp.process(state)

    assert processed_state.shape == processed_shape
