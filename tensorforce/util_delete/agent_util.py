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
Utility functions concerning RL agents.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.config import Config
from tensorforce.exceptions.tensorforce_exceptions import TensorForceValueError
from tensorforce.agents import *


def create_agent(agent_type, config, scope='prefixed_scope'):
    """
    Create agent instance by providing type as a string parameter.

    :param agent_type: String parameter containing agent type
    :param config: Dict containing configuration
    :param scope: Scope prefix used for distributed tensorflow scope separation
    :return: Agent instance
    """
    agent_class = agents.get(agent_type)

    if not agent_class:
        raise TensorForceValueError("No such agent: {}".format(agent_type))

    return agent_class(config, scope)


def get_default_config(agent_type):
    """
    Get default configuration from agent by providing type as a string parameter.

    :param agent_type: String parameter containing agent type
    :return: Default configuration dict
    """
    agent_class = agents.get(agent_type)

    if not agent_class:
        raise TensorForceValueError("No such agent: {}".format(agent_type))

    return Config(agent_class.default_config), Config(agent_class.model_ref.default_config)


agents = {
    'RandomAgent': RandomAgent,
    'DQNAgent': DQNAgent,
    'NAFAgent': NAFAgent,
    'TRPOAgent': TRPOAgent,
    'VPGAgent': VPGAgent,
    'DQFDAgent': DQFDAgent,
}
