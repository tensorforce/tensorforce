# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Comment
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.preprocessing import Preprocessor


class Grayscale(Preprocessor):

    default_config = {
        'weights': [0.299, 0.587, 0.114]
    }

    config_args = [
        'weights'
    ]

    def process(self, state):
        """
        Turn 3D color state into grayscale, thereby removing the last dimension.
        :param state: state input
        :return: new_state
        """
        return (self.config.weights * state).sum(-1)

    def shape(self, original_shape):
        return list(original_shape[:-1])
