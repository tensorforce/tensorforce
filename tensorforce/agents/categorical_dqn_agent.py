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
Categorical DQN agent.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import DQNAgent
from tensorforce.models import CategoricalDQNModel


class CategoricalDQNAgent(DQNAgent):
    """
    Categorical DQN agent. Replaces linear outputs with a distribution for each action as described in
    https://arxiv.org/abs/1707.06887

    Configuration:

    Each agent requires the following configuration parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    The `MemoryAgent` class additionally requires the following parameters:

    * `batch_size`: integer of the batch size.
    * `memory_capacity`: integer of maximum experiences to store.
    * `memory`: string indicating memory type ('replay' or 'prioritized_replay').
    * `update_frequency`: integer indicating the number of steps between model updates.
    * `first_update`: integer indicating the number of steps to pass before the first update.
    * `repeat_update`: integer indicating how often to repeat the model update.

    Each model requires the following configuration parameters:

    * `discount`: float of discount factor (gamma).
    * `learning_rate`: float of learning rate (alpha).
    * `optimizer`: string of optimizer to use (e.g. 'adam').
    * `device`: string of tensorflow device name.
    * `tf_saver`: boolean whether to save model parameters.
    * `tf_summary`: boolean indicating whether to use tensorflow summary file writer.
    * `log_level`: string containing logleve (e.g. 'info').
    * `distributed`: boolean indicating whether to use distributed tensorflow.
    * `global_model`: global model.
    * `session`: session to use.

    The DQN agent expects the following additional configuration parameters:

    * `target_update_frequency`: int of states between updates of the target network.

    The Cateogircal DQN model also expects the following configuration parameters:

    * `distribution_min`: float minimum value of distribution
    * `distribution_max`: float maximum value of distribution
    * `num_atoms`: int number of atoms (discrete steps) to use in the distribution
    * `update_target_weight`: float of update target weight (tau parameter).

    """

    name = 'CategoricalDQNAgent'
    model = CategoricalDQNModel

    def __init__(self, config, model=None):
        config.default(DQNAgent.default_config)
        super(CategoricalDQNAgent, self).__init__(config, model)
