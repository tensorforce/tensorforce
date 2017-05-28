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

import numpy as np
from scipy.misc import imresize

from tensorforce.core.preprocessing.preprocessor import Preprocessor


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
