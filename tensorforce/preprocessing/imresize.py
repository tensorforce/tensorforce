# Copyright 2016 reinforce.io. All Rights Reserved.
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
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorforce.preprocessing.preprocessor import Preprocessor
from scipy.misc import imresize


class Imresize(Preprocessor):

    default_config = {
        'dimensions': [80, 80]
    }

    config_args = [
        'dimensions'
    ]

    def process(self, state):
        """
        Resize image.

        :param state: state input
        :return: new_state
        """
        return imresize(state.astype(np.uint8), self.config.dimensions)

    def shape(self, original_shape):
        return list(original_shape[:-2]) + self.config.dimensions
