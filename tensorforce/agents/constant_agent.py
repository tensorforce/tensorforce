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
Random agent that always returns a random action. Useful to be able to get random
agents with specific shapes.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.models.constant_model import ConstantModel


class ConstantAgent(Agent):
    """
    Constant action agent for sanity checks. Returns a constant value at every
    step, useful to debug continuous problems.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe=1000,
        scope='constant',
        action_values=None
    ):
        """
        Initializes a constant agent which returns a constant action of the provided shape.

        Args:
            action_values: Action value specification, must match actions_spec names
        """

        if action_values is None:
            raise TensorForceError("No action_values for constant model provided.")
        self.action_values = action_values

        super(ConstantAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe,
            scope=scope
        )

    def initialize_model(self):
        return ConstantModel(
            states_spec=self.states_spec,
            actions_spec=self.actions_spec,
            device=None,
            session_config=None,
            scope=self.scope,
            saver_spec=None,
            summary_spec=None,
            distributed_spec=None,
            optimizer=None,
            discount=0.0,
            variable_noise=None,
            states_preprocessing_spec=None,
            explorations_spec=None,
            reward_preprocessing_spec=None,
            action_values=self.action_values
        )

