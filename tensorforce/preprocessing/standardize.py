# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Standardize data (z-transformation)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorforce.preprocessing import Preprocessor
from tensorforce.util.math_util import zero_mean_unit_variance


class Standardize(Preprocessor):

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
        return zero_mean_unit_variance(state.astype(np.float32))
