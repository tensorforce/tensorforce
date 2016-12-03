
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

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.rl_agents.pg_agent import PGAgent
from tensorforce.updater.trpo_updater import TRPOUpdater


class TRPOAgent(PGAgent):

    default_config = {
        'batch_size': 100,
        'update_steps': 10,
        'min_replay_size': 100,
        'deterministic_mode': False
    }

    value_function_ref = TRPOUpdater
