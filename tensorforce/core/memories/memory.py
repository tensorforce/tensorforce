# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from tensorforce import util
from tensorforce.core import Module


class Memory(Module):
    """
    Base class for memories.
    """

    def __init__(
        self, name, states_spec, internals_spec, actions_spec, include_next_states,
        summary_labels=None
    ):
        """
        Args:
            state_spec (dict): State specification.
            internals_spec (dict): Internal state specification.
            action_spec (dict): Action specification.
            include_next_states (bool): Include subsequent state if true.
        """
        super().__init__(name=name, l2_regularization=0.0, summary_labels=summary_labels)

        self.states_spec = states_spec
        self.internals_spec = internals_spec
        self.actions_spec = actions_spec
        self.include_next_states = include_next_states

    def tf_store(self, states, internals, actions, terminal, reward):
        """"
        Stores experiences, i.e. a batch of timesteps.

        Args:
            state: Dict of state tensors.
            internal: List of prior internal state tensors.
            action: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
        """
        raise NotImplementedError

    def tf_retrieve_timesteps(self, n):
        """
        Retrieves a given number of timesteps from the stored experiences.

        Args:
            n: Number of timesteps to retrieve.

        Returns:
            Dicts containing the retrieved experiences.
        """
        raise NotImplementedError

    def tf_retrieve_episodes(self, n):
        """
        Retrieves a given number of episodes from the stored experiences.

        Args:
            n: Number of episodes to retrieve.

        Returns:
            Dicts containing the retrieved experiences.
        """
        raise NotImplementedError

    def tf_retrieve_sequences(self, n, sequence_length):
        """
        Retrieves a given number of temporally consistent timestep sequences from the stored
        experiences.

        Args:
            n: Number of sequences to retrieve.
            sequence_length: Length of timestep sequences.

        Returns:
            Dicts containing the retrieved experiences.
        """
        raise NotImplementedError

    def tf_update_batch(self, loss_per_instance):
        """
        Updates the internal information of the latest batch instances based on their loss.

        Args:
            loss_per_instance: Loss per instance tensor.
        """
        return util.no_operation()
