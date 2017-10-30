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

from tensorforce import util
import tensorforce.core.memories


# TODO: implement in TensorFlow

class Memory(object):
    """
    Abstract memory class.
    """

    def __init__(self, states_spec, actions_spec):
        """
        Generic memory without sampling strategy implemented.

        Args:
            states_spec: State specifiction
            actions_spec: Action specification
        """
        self.states_spec = states_spec
        self.actions_spec = actions_spec

    def add_observation(self, states, internals, actions, terminal, reward):
        """
        Inserts a single experience to the memory.

        Args:
            states:
            internals:
            actions:
            terminal:
            reward:

        Returns:

        """
        raise NotImplementedError

    def get_batch(self, batch_size, next_states=False):
        """
        Samples a batch from the memory.

        Args:
            batch_size: The batch size
            next_states: A boolean flag indicating whether 'next_states' values should be included

        Returns: A dict containing states, internal states, actions, terminals, rewards (and next states)

        """
        raise NotImplementedError

    def update_batch(self, loss_per_instance):
        """
        Updates loss values for sampling strategies based on loss functions.

        Args:
            loss_per_instance:

        """
        raise NotImplementedError

    def set_memory(self, states, internals, actions, terminals, rewards):
        """
        Deletes memory content and sets content to provided observations.

        Args:
            states:
            internals:
            actions:
            terminals:
            rewards:

        """
        raise NotImplementedError

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a memory from a specification dict.
        """
        memory = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.memories.memories,
            kwargs=kwargs
        )
        assert isinstance(memory, Memory)
        return memory
