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

    default_config = dict(
        # Agent
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        # Logging
        log_level='info',
        model_directory=None,
        save_frequency=600,  # TensorFlow default
        summary_labels=['total-loss'],
        summary_frequency=120,  # TensorFlow default
        # TensorFlow distributed configuration
        cluster_spec=None,
        parameter_server=False,
        task_index=0,
        device=None,
        local_model=False,
        replica_model=False,
        scope='random'
    )

    def __init__(self, states_spec, actions_spec, config):
        config = config.copy()
        config.default(self.__class__.default_config)
        config.obligatory(
            optimizer=None,
            discount=1.0,
            normalize_rewards=False,
            variable_noise=None
        )
        super(RandomAgent, self).__init__(states_spec, actions_spec, config)

    def initialize_model(self, states_spec, actions_spec, config):
        return RandomModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            config=config
        )
