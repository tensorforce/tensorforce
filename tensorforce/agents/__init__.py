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

from tensorforce.agents.dqn_agent import DQNAgent
from tensorforce.agents.naf_agent import NAFAgent
from tensorforce.agents.vpg_agent import VPGAgent
from tensorforce.agents.trpo_agent import TRPOAgent
from tensorforce.agents.dqfd_agent import DQFDAgent


agents = dict(
    DQNAgent=DQNAgent,
    NAFAgent=NAFAgent,
    VPGAgent=VPGAgent,
    TRPOAgent=TRPOAgent,
    DQFDAgent=DQFDAgent
)


__all__ = ['DQNAgent', 'NAFAgent', 'VPGAgent', 'TRPOAgent', 'DQFDAgent', 'agents']
