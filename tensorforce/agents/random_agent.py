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

from tensorforce.agents import Agent
from tensorforce.models.random_model import RandomModel


class RandomAgent(Agent):
    """
    Random agent, useful as a baseline and sanity check.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe=1000,
        scope='random',
    ):
        """
        Initializes a random agent. Returns random actions based of the shape
        provided in the 'actions_spec'.

        """

        super(RandomAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe,
            scope=scope
        )

    def initialize_model(self):
        return RandomModel(
            states_spec=self.states_spec,
            actions_spec=self.actions_spec,
            device=None,
            session_config=None,
            scope=self.scope,
            saver_spec=None,
            summary_spec=None,  # TODO: remove from RandomModel or make Model c'tor more flexible (add default values)
            distributed_spec=None,
            optimizer=None,
            discount=0.0,  # TODO: remove from RandomModel or make Model c'tor more flexible (add default values)
            variable_noise=None,
            states_preprocessing_spec=None,
            explorations_spec=None,
            reward_preprocessing_spec=None
        )
