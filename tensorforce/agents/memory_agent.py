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

from tensorforce import util
from tensorforce.agents import Agent
from tensorforce.core.memories import memories


class MemoryAgent(Agent):

    default_config = dict(
        batch_size=1,
        memory_capacity=1000000,
        memory='replay',
        memory_args=None,
        memory_kwargs=None,
        update_frequency=4,
        first_update=10000,
        repeat_update=1
    )

    def __init__(self, config):
        config.default(MemoryAgent.default_config)
        super(MemoryAgent, self).__init__(config)

        self.batch_size = config.batch_size
        memory = util.function(config.memory, memories)
        args = config.memory_args or ()
        kwargs = config.memory_kwargs or {}
        self.memory = memory(config.memory_capacity, config.states, config.actions, *args, **kwargs)
        self.update_frequency = config.update_frequency
        self.first_update = config.first_update
        self.repeat_update = config.repeat_update

    def observe(self, reward, terminal):
        self.current_reward = reward
        self.current_terminal = terminal

        self.memory.add_experience(state=self.current_state, action=self.current_action, reward=self.current_reward, terminal=self.current_terminal, internal=self.current_internal)

        if self.timestep >= self.first_update and self.timestep % self.update_frequency == 0:
            for _ in xrange(self.repeat_update):
                batch = self.memory.get_batch(batch_size=self.batch_size)
                _, loss_per_instance = self.model.update(batch=batch)
                self.memory.update_batch(loss_per_instance=loss_per_instance)

    def update_batch(self, loss_per_instance):
        pass
