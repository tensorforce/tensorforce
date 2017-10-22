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

import numpy as np

from tensorforce.core.preprocessing import Preprocessor


class Grayscale(Preprocessor):
    """
    Turn 3D color state into grayscale.
    """

    def __init__(self, weights=(0.299, 0.587, 0.114)):
        super(Grayscale, self).__init__()
        self.weights = weights

    def process(self, state):
        state = (self.weights * state).sum(-1)
        return np.reshape(state, tuple(state.shape) + (1,))

    def processed_shape(self, shape):
        return tuple(shape[:-1]) + (1,)
