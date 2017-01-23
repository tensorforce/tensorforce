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
