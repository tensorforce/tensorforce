# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from tensorforce import TensorforceError


class MultiplayerEnvironment(object):
    """
    Multi-player environment base class.
    """

    def __init__(self, num_players):
        self.num_players = num_players

    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are 
        available simultaneously.

        Returns:
            States specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (default: 'float').
                - shape: integer, or list/tuple of integers (required).
        """
        raise NotImplementedError

    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are 
        available simultaneously.

        Returns:
            actions (spec, or dict of specs): Actions specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (required).
                - shape: integer, or list/tuple of integers (default: []).
                - num_actions: integer (required if type == 'int').
                - min_value and max_value: float (optional if type == 'float', default: none).
        """
        raise NotImplementedError

    def close(self):
        pass

    def reset(self):
        # no return
        raise NotImplementedError

    def get_state(self, player):
        # return state for player
        raise NotImplementedError

    def execute(self, actions, player):
        # return terminal, reward
        raise NotImplementedError
