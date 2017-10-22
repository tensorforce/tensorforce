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


class Sequence(Preprocessor):
    """
    Concatenate `length` state vectors. Example: Used in Atari
    problems to create the Markov property.
    """

    def __init__(self, length=2):
        super(Sequence, self).__init__()
        self.length = length
        self.index = -1

    def process(self, state):
        if self.index == -1:
            self.previous_states = [state for _ in range(self.length)]
            self.index = 1
        else:
            self.previous_states[self.index % self.length] = state
            self.index += 1
        sequence = [self.previous_states[n % self.length] for n in range(self.index, self.index + self.length)]
        return np.concatenate(sequence, -1)

    def processed_shape(self, shape):
        return shape[:-1] + (shape[-1] * self.length,)

    def reset(self):
        self.index = -1
