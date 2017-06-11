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

from six.moves import xrange

from tensorforce.core import Agent
from tensorforce.core.memories import ReplayMemory


class MemoryAgent(Agent):

    default_config = dict(
        batch_size=32,
        memory_capacity=1000000,
        update_frequency=4,
        first_update=10000,
        repeat_update=1
    )

    def __init__(self, config):
        config.default(MemoryAgent.default_config)
        super(MemoryAgent, self).__init__(config)

        self.batch_size = config.batch_size
        self.memory = ReplayMemory(capacity=config.memory_capacity, states_config=config.states, actions_config=config.actions)
        self.update_frequency = config.update_frequency
        self.first_update = config.first_update
        self.repeat_update = config.repeat_update

    def observe(self, state, action, reward, terminal):
        if self.unique_state:
            state = dict(state=state)
        if self.unique_action:
            action = dict(action=action)

        self.memory.add_experience(state=state, action=action, reward=reward, terminal=terminal, internal=self.internals)

        if self.timestep >= self.first_update and self.timestep % self.update_frequency == 0:
            for _ in xrange(self.repeat_update):
                batch = self.memory.get_batch(batch_size=self.batch_size)
                self.model.update(batch=batch)
