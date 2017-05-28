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


class Maximum(Preprocessor):

    default_config = {
        'count': 2
    }

    config_args = [
        'count'
    ]

    def __init__(self, config, *args, **kwargs):
        super(Maximum, self).__init__(config, *args, **kwargs)

        self._queue = deque(maxlen=self.config.count)

    def process(self, state):
        """
        Returns maximum of states over the last self.config.count states
        :param state: state input
        :return: new_state
        """
        self._queue.append(state)

        # If queue is too short, fill with current state.
        while len(self._queue) < self.config.count:
            self._queue.append(state)

        return np.max(np.array(self._queue), axis=0)
