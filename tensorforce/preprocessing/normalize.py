# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Normalize data by rescaling.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorforce.preprocessing import Preprocessor
from tensorforce.util.math_util import unity_based_normalization


class Normalize(Preprocessor):

    default_config = {
    }

    config_args = [
    ]

    def process(self, state):
        """
        Standardize the data.
        :param state: state input
        :return: new_state
        """
        return unity_based_normalization(state.astype(np.float32))
