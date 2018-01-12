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
    which it samples batches according to some sampling strategy to
    update the value function.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe,
        batch_size,
        memory,
        first_update,
        update_frequency,
        repeat_update
    ):
        """

        Args:
            states_spec: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions_spec: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
            batch_size: Int specifying batch size used to sample from memory. Should be smaller than memory size.
            memory: Dict describing memory via `type` (e.g. `replay`) and `capacity`.
            first_update: Int describing at which time step the first update is performed. Should be larger
                than batch size.
            update_frequency: Int specifying number of observe steps to perform until an update is executed.
            repeat_update: Int specifying how many update steps are performed per update, where each update step implies
                sampling a batch from the memory and passing it to the model.
        """
        self.memory_spec = memory
        self.batch_size = batch_size
        self.first_update = first_update
        self.update_frequency = update_frequency
        self.repeat_update = repeat_update

        super(MemoryAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
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
