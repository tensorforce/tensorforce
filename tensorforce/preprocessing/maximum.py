# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Comment
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import deque

import numpy as np

from tensorforce.preprocessing import Preprocessor


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
        while len(self._queue) < self.config.concat_length:
            self._queue.append(state)

        return np.max(np.array(self._queue), axis=0)
