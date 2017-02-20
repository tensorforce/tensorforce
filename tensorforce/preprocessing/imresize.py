# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Comment
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorforce.preprocessing import Preprocessor
from scipy.misc import imresize


class Imresize(Preprocessor):

    default_config = {
        'dimension_x': 84,
        'dimension_y': 84
    }

    config_args = [
        'dimension_x',
        'dimension_y'
    ]

    def process(self, state):
        """
        Resize image.

        :param state: state input
        :return: new_state
        """
        return imresize(state.astype(np.uint8), [self.config.dimension_x, self.config.dimension_y])

    def shape(self, original_shape):
        return original_shape[:-2] + [self.config.dimension_x, self.config.dimension_y]
