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

from tensorforce.agents.learning_agent import LearningAgent
from tensorforce.core.memories import Memory


class MemoryAgent(LearningAgent):
    """
    The `MemoryAgent` class implements a replay memory from
    which it samples batches according to some sampling strategy to
    update the value function.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe=1000,
        scope='memory_agent',
        # parameters specific to LearningAgents
        summary_spec=None,
        network_spec=None,
        discount=0.99,
        device=None,
        session_config=None,
        saver_spec=None,
        distributed_spec=None,
        optimizer=None,
        variable_noise=None,
        states_preprocessing_spec=None,
        explorations_spec=None,
        reward_preprocessing_spec=None,
        distributions_spec=None,
        entropy_regularization=None,
        # parameters specific to MemoryAgents
        batch_size=1000,
        memory=None,
        first_update=10000,
        update_frequency=4,
        repeat_update=1
    ):
        """

        Args:
            batch_size (int): The batch size used to sample from memory. Should be smaller than memory size.
            memory (Union[dict,Memory]): Dict describing memory via `type` (e.g. `replay`) and `capacity`.
                Alternatively, an actual Memory object can be passed in directly.
            first_update (int): At which time step the first update is performed. Should be larger
                than batch size.
            update_frequency (int): Number of `observe` steps to perform until an update is executed.
            repeat_update (int): How many update steps are performed per update, where each update step implies
                sampling a batch from the memory and passing it to the model.
        """
        super(MemoryAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe,
            scope=scope,
            # parameters specific to LearningAgent
            summary_spec=summary_spec,
            network_spec=network_spec,
            discount=discount,
            device=device,
            session_config=session_config,
            saver_spec=saver_spec,
            distributed_spec=distributed_spec,
            optimizer=optimizer,
            variable_noise=variable_noise,
            states_preprocessing_spec=states_preprocessing_spec,
            explorations_spec=explorations_spec,
            reward_preprocessing_spec=reward_preprocessing_spec,
            distributions_spec=distributions_spec,
            entropy_regularization=entropy_regularization
        )

        # Memory already given as a Memory object: Use that.
        if isinstance(memory, Memory):
            self.memory = memory
            self.memory_spec = None
        else:
            # Nothing given: Create a default memory spec.
            if memory is None:
                memory = dict(
                    type='replay',
                    capacity=100000
                )
            # Now create actual Memory object from the spec.
            self.memory_spec = memory
            self.memory = Memory.from_spec(
                spec=self.memory_spec,
                kwargs=dict(
                    states_spec=self.states_spec,
                    actions_spec=self.actions_spec
                )
            )
        self.batch_size = batch_size
        self.first_update = first_update
        self.update_frequency = update_frequency
        self.repeat_update = repeat_update

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
                    states={name: np.stack((batch['states'][name],
                                            batch['next_states'][name])) for name in batch['states']},
                    internals=batch['internals'],
                    actions=batch['actions'],
                    terminal=batch['terminal'],
                    reward=batch['reward'],
                    return_loss_per_instance=True
                )
                self.memory.update_batch(loss_per_instance=loss_per_instance)

    def import_observations(self, observations):
        """
        Load an iterable of observation dicts into the replay memory.

        Args:
            observations: An iterable with each element containing an observation. Each
                observation requires keys 'state','action','reward','terminal', 'internal'.
                Use an empty list [] for 'internal' if internal state is irrelevant.
        """
        for observation in observations:
            self.memory.add_observation(
                states=observation['states'],
                internals=observation['internals'],
                actions=observation['actions'],
                terminal=observation['terminal'],
                reward=observation['reward']
            )
