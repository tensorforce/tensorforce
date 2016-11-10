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
Implements normalized advantage functions as described here:
https://arxiv.org/abs/1603.00748
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.value_functions.value_function import ValueFunction


class NormalizedAdvantageFunctions(ValueFunction):

    default_config = {
        'tau': 0,
        'epsilon': 0.1,
        'gamma': 0,
        'alpha': 0.5,
        'clip_gradients': False
    }

    def __init__(self, config):
        """
        Training logic for NAFs.

        :param config: Configuration parameters
        """
        super(NormalizedAdvantageFunctions, self).__init__(config)


    def create_outputs(self):
        """
        We use get_network to define the hidden layers and create the NAF specific
        outputs in this function.
        """
        pass

    def create_training_operations(self):
        """
        NAF training logic.
        """

        pass
