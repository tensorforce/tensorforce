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
Base environment class
"""

class Environment(object):

    def reset(self):
        """
        Reset environment and setup for new episode

        :return:
        """
        raise NotImplementedError

    def execute_action(self, action):
        """
        Executes action, observes next state and reward

        :param action: Action to execute

        :return: dict containing next_state, reward, and boolean indicating
            if next state is a terminal state
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """
        Get action space

        :return: Object of type tensorforce.spaces.space.Space containing the action space
        """
        raise NotImplementedError

    @property
    def state_space(self):
        """
        Return state space

        :return: Object of type tensorforce.spaces.space.Space containing the state space
        """
        raise NotImplementedError
