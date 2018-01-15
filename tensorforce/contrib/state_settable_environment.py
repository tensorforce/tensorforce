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

from tensorforce.environments import Environment


class StateSettableEnvironment(Environment):
    """
    An Environment that implements the set_state method to set the current state
    to some new state using setter instructions.
    """
    def set_state(self, **kwargs):
        """
        Sets the current state of the environment manually to some other state and returns a new observation.

        Args:
            **kwargs: The set instruction(s) to be executed by the environment.
                       A single set instruction usually set a single property of the
                      state/observation vector to some new value.
        Returns: The observation dictionary of the Environment after(!) setting it to the new state.
        """
        raise NotImplementedError

