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
Vanilla policy gradient agent.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import BatchAgent
from tensorforce.models import VPGModel


class VPGAgent(BatchAgent):
    """
    Vanilla Policy Gradient agent as described by [Sutton et al. (1999)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf).

    Configuration:

    Each agent requires the following ``Configuration`` parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    The `BatchAgent` class additionally requires the following parameters:

    * `batch_size`: integer of the batch size.

    A Policy Gradient Model expects the following additional configuration parameters:

    * `baseline`: string indicating the baseline value function (currently 'linear' or 'mlp').
    * `generalized_advantage_estimation`: boolean indicating whether to use GAE.
    * `gae_lambda`: float of the Generalized Advantage Estimation lambda.
    * `normalize_advantage`: boolean indicating whether to normalize the advantage or not.

    The VPG agent does not require any additional configuration parameters.

    """


    name = 'VPGAgent'
    model = VPGModel
