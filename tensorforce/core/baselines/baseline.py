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
Generic baseline value function.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce import util
import tensorforce.core.baselines


class Baseline(object):

    def create_tf_operations(self, state, batch_size, scope=''):
        raise NotImplementedError

    def predict(self, states):
        """Predicts the state-value function V(s)

        Args:
            states: State or batch of states

        Returns: V(s)

        """
        raise NotImplementedError

    def update(self, states, returns):
        """
        Fits baseline to returns.

        Args:
            states: State or batch of states
            returns: Returns for states

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def from_config(config):
        """
        Creates a baseline from a configuration dict.

        Args:
            config:

        Returns:

        """
        return util.get_object(
            obj=config,
            predefined=tensorforce.core.baselines.baselines
        )
