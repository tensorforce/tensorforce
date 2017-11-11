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

import numpy as np

from tensorforce.agents import Agent
from tensorforce.core.memories import Memory


class MemoryAgent(Agent):
    """
    The `MemoryAgent` class implements a replay memory from
    which it samples batches to update the value function.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        preprocessing,
        exploration,
        reward_preprocessing,
        batched_observe,
        batch_size,
        memory,
        first_update,
        update_frequency,
        repeat_update
    ):
        self.memory_spec = memory
        self.batch_size = batch_size
        self.first_update = first_update
        self.update_frequency = update_frequency
        self.repeat_update = repeat_update

        super(MemoryAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            preprocessing=preprocessing,
            exploration=exploration,
            reward_preprocessing=reward_preprocessing,
            batched_observe=batched_observe
        )

        self.memory = Memory.from_spec(
            spec=self.memory_spec,
            kwargs=dict(
                states_spec=self.states_spec,
                actions_spec=self.actions_spec
            )
        )

    def observe(self, terminal, reward):
        super(MemoryAgent, self).observe(terminal=terminal, reward=reward)

        self.memory.add_observation(
            states=self.current_states,
            internals=self.current_internals,
            actions=self.current_actions,
            terminal=self.current_terminal,
            reward=self.current_reward
        )

        if self.timestep >= self.first_update and self.timestep % self.update_frequency == 0:

            for _ in xrange(self.repeat_update):
                batch = self.memory.get_batch(batch_size=self.batch_size, next_states=True)
                loss_per_instance = self.model.update(
                    # TEMP: Random sampling fix
                    states={name: np.stack((batch['states'][name], batch['next_states'][name])) for name in batch['states']},
                    internals=batch['internals'],
                    actions=batch['actions'],
                    terminal=batch['terminal'],
                    reward=batch['reward'],
                    return_loss_per_instance=True
                )
                self.memory.update_batch(loss_per_instance=loss_per_instance)

    def import_observations(self, observations):
        """Load an iterable of observation dicts into the replay memory.

        Args:
            observations: An iterable with each element containing an observation. Each
            observation requires keys 'state','action','reward','terminal', 'internal'.
            Use an empty list [] for 'internal' if internal state is irrelevant.

        Returns:

        """
        for observation in observations:
            self.memory.add_observation(
                states=observation['states'],
                internals=observation['internals'],
                actions=observation['actions'],
                terminal=observation['terminal'],
                reward=observation['reward']
            )
