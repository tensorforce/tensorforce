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
    """

    default_config = dict(
        batch_size=1,
        memory_capacity=1e5,
        memory=dict(
            type='replay',
            random_sampling=True
        ),
        update_frequency=4,
        first_update=10000,
        repeat_update=1
    )

    def __init__(self, states_spec, actions_spec, config):
        config.default(MemoryAgent.default_config)
        self.batch_size = config.batch_size
        self.memory_capacity = config.memory_capacity
        self.update_frequency = config.update_frequency
        self.first_update = config.first_update
        self.repeat_update = config.repeat_update

        super(MemoryAgent, self).__init__(states_spec, actions_spec, config)

        self.memory = Memory.from_spec(
            spec=config.memory,
            kwargs=dict(
                capacity=self.memory_capacity,
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
                loss_per_instance = self.model.update(batch=batch, return_loss_per_instance=True)
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
