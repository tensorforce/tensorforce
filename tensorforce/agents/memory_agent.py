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

from tensorforce.agents import Agent
from tensorforce.core.memories import Memory


class MemoryAgent(Agent):
    """
    The `MemoryAgent` class implements a replay memory, from which it samples batches to update the value function.

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

    """

    default_config = dict(
        batch_size=1,
        memory_capacity=1000000,
        memory='replay',
        update_frequency=4,
        first_update=10000,
        repeat_update=1
    )

    def __init__(self, config):
        config.default(MemoryAgent.default_config)
        super(MemoryAgent, self).__init__(config)

        self.batch_size = config.batch_size
        self.memory = Memory.from_config(
            config=config.memory,
            kwargs=dict(
                capacity=config.memory_capacity,
                states_config=config.states,
                actions_config=config.actions
            )
        )
        self.update_frequency = config.update_frequency
        self.first_update = config.first_update
        self.repeat_update = config.repeat_update

    def observe(self, reward, terminal):
        self.current_reward = reward
        self.current_terminal = terminal

        self.memory.add_observation(
            state=self.current_state,
            action=self.current_action,
            reward=self.current_reward,
            terminal=self.current_terminal,
            internal=self.current_internal
        )

        if self.timestep >= self.first_update and self.timestep % self.update_frequency == 0:
            for _ in xrange(self.repeat_update):
                batch = self.memory.get_batch(batch_size=self.batch_size)
                _, loss_per_instance = self.model.update(batch=batch)
                self.memory.update_batch(loss_per_instance=loss_per_instance)

    def import_observations(self, observations):
        for observation in observations:
            self.memory.add_observation(
                state=observation['state'],
                action=observation['action'],
                reward=observation['reward'],
                terminal=observation['terminal'],
                internal=observation['internal']
            )
