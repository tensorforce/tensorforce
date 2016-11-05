# Copyright 2016 reinforce.io. All Rights Reserved.
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

from tensorforce.exceptions.tensorforce_exceptions import TensorForceValueError
from tensorforce.rl_agents import *


def create_agent(type, agent_config, network_config):
    """
    Create agent instance by providing type as a string parameter.

    :param type: String parameter containing agent type
    :param agent_config: Dict containing agent configuration
    :param network_config: Dict containing network configuration
    :return: Agent instance
    """
    agent_class = agents.get(type)

    if not agent_class:
        raise TensorForceValueError("No such agent: {}".format(type))

    return agent_class(agent_config, network_config)


agents = {
    'RandomAgent': RandomAgent,
    'DQNAgent': DQNAgent
}