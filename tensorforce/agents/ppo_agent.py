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

from tensorforce.agents import BatchAgent
from tensorforce.models.ppo_model import PPOModel


class PPOAgent(BatchAgent):
    """
    Proximal Policy Optimization agent ([Schulman et al., 2017]
    (https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf).

    Configuration:

    Each agent requires the following ``Configuration`` parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    The `BatchAgent` class additionally requires the following parameters:

    * `batch_size`: integer of the batch size.
    * `keep_last`: bool optionally keep the last observation for use in the next batch

    A Policy Gradient Model expects the following additional configuration parameters:

    * `baseline`: string indicating the baseline value function (currently 'linear' or 'mlp').
    * `baseline_args`: list of arguments for the baseline value function.
    * `baseline_kwargs`: dict of keyword arguments for the baseline value function.
    * `generalized_advantage_estimation`: boolean indicating whether to use GAE estimation.
    * `gae_lambda`: float of the Generalized Advantage Estimation lambda.
    * `normalize_advantage`: boolean indicating whether to normalize the advantage or not.


    The TRPO agent expects the following additional configuration parameters:

    * `learning_rate`: float of learning rate (alpha).
    * `optimizer`: string of optimizer to use (e.g. 'adam').


    """

    name = 'PPOAgent'
    model = PPOModel
