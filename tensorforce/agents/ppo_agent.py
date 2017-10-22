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
Proximal Policy Optimization agent.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import MemoryAgent
from tensorforce.models.ppo_model import PPOModel

class PPOAgent(MemoryAgent):
    """
    Proximal Policy Optimization agent ([Schulman et al., 2017]
    (https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf).

    Configuration:

    Each agent requires the following ``Configuration`` parameters:

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
    * `tf_summary`: string directory to write tensorflow summaries. Default None
    * `tf_summary_level`: int indicating which tensorflow summaries to create.
    * `tf_summary_interval`: int number of calls to get_action until writing tensorflow summaries on update.
    * `log_level`: string containing loglevel (e.g. 'info').
    * `distributed`: boolean indicating whether to use distributed tensorflow.
    * `global_model`: global model.
    * `session`: session to use.

    A Policy Gradient Model expects the following additional configuration parameters:

    * `baseline`: string indicating the baseline value function (currently 'linear' or 'mlp').
    * `baseline_args`: list of arguments for the baseline value function.
    * `baseline_kwargs`: dict of keyword arguments for the baseline value function.
    * `gae_rewards`: boolean indicating whether to use GAE reward estimation.
    * `gae_lambda`: GAE lambda.
    * `normalize_rewards`: boolean indicating whether to normalize rewards.

    The PPO agent expects the following additional configuration parameters:

    * `entropy_penalty`: float (e.g. 0.01)
    * `loss_clipping`: float trust region clipping (e.g. 0.2)
    * `epochs`: int number of training epochs for optimizer (e.g. 10)
    * `optimizer_batch_size`: int batch size for optimizer (e.g. 64)

    """

    name = 'PPOAgent'
    model = PPOModel

    default_config = dict(
        batch_size=128,
        memory=dict(
            type='prioritized_replay',
        ),
        update_frequency=128,
        first_update=256,
    )

    def __init__(self, config, model=None):
        config.default(PPOAgent.default_config)
        super(PPOAgent, self).__init__(config, model)
