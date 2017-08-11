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
NStep Batched DQN agent.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import BatchAgent
from tensorforce.models import DQNNstepModel


class DQNNstepAgent(BatchAgent):
    """
    Nstep Deep-Q-Network agent (DQN). Uses the last experiences (batch)
    to train on.

    DQN chooses from one of a number of discrete actions by taking the maximum Q-value
    from the value function with one output neuron per available action.

    Configuration:

    Each agent requires the following configuration parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    The `BatchAgent` class additionally requires the following parameters:

    * `batch_size`: integer of the batch size.
    * `keep_last`: bool optionally keep the last observation for use in the next batch

    Each model requires the following configuration parameters:

    * `discount`: float of discount factor (gamma).
    * `learning_rate`: float of learning rate (alpha).
    * `optimizer`: string of optimizer to use (e.g. 'adam').
    * `device`: string of tensorflow device name.
    * `tf_summary`: string directory to write tensorflow summaries. Default None
    * `tf_summary_level`: int indicating which tensorflow summaries to create.
    * `tf_summary_interval`: int number of calls to get_action until writing tensorflow summaries on update.
    * `log_level`: string containing logleve (e.g. 'info').
    * `distributed`: boolean indicating whether to use distributed tensorflow.
    * `global_model`: global model.
    * `session`: session to use.

    The DQN Nstep agent expects the following additional configuration parameters:

    * `target_update_frequency`: int of states between updates of the target network.
    * `update_target_weight`: float of update target weight (tau parameter).
    * `double_dqn`: boolean indicating whether to use double-dqn.
    * `clip_loss`: float if not 0, uses the huber loss with clip_loss as the linear bound

    """

    name = 'DQNNstepAgent'
    model = DQNNstepModel
