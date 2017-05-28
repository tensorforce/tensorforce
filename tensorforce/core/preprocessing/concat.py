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
Comment
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque

import numpy as np

from tensorforce.core.preprocessing.preprocessor import Preprocessor


class Concat(Preprocessor):

    default_config = {
        'concat_length': 1,
        'dimension_position': 'prepend'
    }

    config_args = [
        'concat_length',
        'dimension_position'
    ]

    def __init__(self, config, *args, **kwargs):
        super(Concat, self).__init__(config, *args, **kwargs)

        self._queue = deque(maxlen=self.config.concat_length)

    def process(self, state):
        """
        Return full concatenated state including new state state.

        :param state: New state to be added
        :return: State tensor of shape (concat_length, state_shape)
        """
        self._queue.append(state)

        # If queue is too short, fill with current state.
        while len(self._queue) < self.config.concat_length:
            self._queue.append(state)

        if self.config.dimension_position == 'append':
            concatted = np.array(self._queue)
            return np.moveaxis(concatted, 0, -1)
        else:
            return np.array(self._queue)

    def shape(self, original_shape):
        if self.config.dimension_position == 'append':
            return list(original_shape) + [self.config.concat_length]
        else:
            return [self.config.concat_length] + list(original_shape)
