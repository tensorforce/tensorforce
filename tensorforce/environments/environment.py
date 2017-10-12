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


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class Environment(object):
    """
    Base environment class.
    """

    def __str__(self):
        raise NotImplementedError

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns: initial state of resetted environment.
        """
        raise NotImplementedError

    def execute(self, action):
        """
        Executes action, observes next state and reward.

        Args:
            action: Action to execute.

        Returns: tuple of state (tuple), terminal_state (bool), and reward (float).
        """
        raise NotImplementedError

    @property
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are available simultaneously.

        Returns: dict of state properties (shape and type).

        """
        raise NotImplementedError

    @property
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are available simultaneously.

        Returns: dict of action properties (continuous, number of actions)

        """
        raise NotImplementedError
