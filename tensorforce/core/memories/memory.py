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

import tensorflow as tf

from tensorforce import util
import tensorforce.core.memories


class Memory(object):
    """
    Base class for memories. A Memory stores records of the type: "state/action/reward/(next-state)?/is-terminal".
    """

    def __init__(self, states, internals, actions, include_next_states, scope='memory', summary_labels=None):
        """
        Args:
            states (dict): States specification.
            internals (dict): Internal states specification.
            actions (dict): Actions specification.
            include_next_states (bool): Include subsequent state if true.
            scope (str): The tf variable scope to use when creating variables for this memory.
            summary_labels (list): List of summary labels.
        """
        self.states_spec = states
        self.internals_spec = internals
        self.actions_spec = actions
        self.include_next_states = include_next_states
        self.scope = scope
        self.summary_labels = set(summary_labels or ())

        self.variables = dict()

        # TensorFlow functions.
        self.initialize = None  # type: callable
        self.store = None
        self.retrieve_timesteps = None
        self.retrieve_episodes = None
        self.retrieve_sequences = None
        self.update_batch = None

        self.setup_template_funcs()

    def setup_template_funcs(self, custom_getter=None):
        if custom_getter is None:
            def custom_getter(getter, name, registered=False, **kwargs):
                variable = getter(name=name, registered=True, **kwargs)
                if registered:
                    pass
                elif name in self.variables:
                    assert variable is self.variables[name]
                else:
                    assert not kwargs['trainable']
                    self.variables[name] = variable
                return variable

        self.initialize = tf.make_template(
            name_=(self.scope + '/initialize'),
            func_=self.tf_initialize,
            custom_getter_=custom_getter
        )
        self.store = tf.make_template(
            name_=(self.scope + '/store'),
            func_=self.tf_store,
            custom_getter_=custom_getter
        )
        self.retrieve_timesteps = tf.make_template(
            name_=(self.scope + '/retrieve_timesteps'),
            func_=self.tf_retrieve_timesteps,
            custom_getter_=custom_getter
        )
        self.retrieve_episodes = tf.make_template(
            name_=(self.scope + '/retrieve_episodes'),
            func_=self.tf_retrieve_episodes,
            custom_getter_=custom_getter
        )
        self.retrieve_sequences = tf.make_template(
            name_=(self.scope + '/retrieve_sequences'),
            func_=self.tf_retrieve_sequences,
            custom_getter_=custom_getter
        )
        self.update_batch = tf.make_template(
            name_=(self.scope + '/update_batch'),
            func_=self.tf_update_batch,
            custom_getter_=custom_getter
        )

        return custom_getter

    def tf_initialize(self):
        """
        Initializes the memory. Called by a memory-model in its own tf_initialize method.
        """
        raise NotImplementedError

    def tf_store(self, states, internals, actions, terminal, reward):
        """"
        Stores experiences, i.e. a batch of timesteps.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
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
        return tf.no_op()

    def get_variables(self):
        """
        Returns the TensorFlow variables used by the memory.

        Returns:
            List of variables.
        """
        return [self.variables[key] for key in sorted(self.variables)]

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
