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
Preprocessor base class
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.config import create_config

class Preprocessor(object):

    # dict containing default configuration
    default_config = {}

    # list specifying order of *args to be parsed
    config_args = []

    def __init__(self, *args, **kwargs):
        """
        Initialize configuration using the default config. Then update the config first using *args (order is
        defined in self.config_args) and then using **kwargs)

        :param args: optional *args
        :param kwargs: optional **kwargs
        """
        self.config = create_config([], default=self.default_config)

        for i, arg in enumerate(args):
            if i >= len(self.config_args):
                break
            self.config.update({self.config_args[i]: arg})

        self.config.update(kwargs)


    def process(self, state):
        """
        Process state.

        :param state: ndarray
        :return: new_state
        """
        return state

    def shape(self, original_shape):
        """
        Return shape of processed state given original shape

        :param original_shape: original shape array
        :return: new shape array
        """
        return original_shape
