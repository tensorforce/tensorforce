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

from copy import deepcopy

import numpy as np

from tensorforce import util, TensorForceError
import tensorforce.agents
from tensorforce.contrib.sanity_check_specs import sanity_check_states, sanity_check_actions


class Agent(object):
    """
    Base class for TensorForce agents.
    """

    def __init__(
        self,
        states,
        actions,
        batched_observe=True,
        batching_capacity=1000
    ):
        """
        Initializes the agent.

        Args:
            states (spec, or dict of specs): States specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (default: 'float').
                - shape: integer, or list/tuple of integers (required).
            actions (spec, or dict of specs): Actions specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (required).
                - shape: integer, or list/tuple of integers (default: []).
                - num_actions: integer (required if type == 'int').
                - min_value and max_value: float (optional if type == 'float', default: none).
            batched_observe (bool): Specifies whether calls to model.observe() are batched, for
                improved performance (default: true).
            batching_capacity (int): Batching capacity of agent and model (default: 1000).
        """

        self.states, self.unique_state = sanity_check_states(states)
        self.actions, self.unique_action = sanity_check_actions(actions)

        # Batched observe for better performance with Python.
        self.batched_observe = batched_observe
        self.batching_capacity = batching_capacity

        self.current_states = None
        self.current_actions = None
        self.current_internals = None
        self.next_internals = None
        self.current_terminal = None
        self.current_reward = None
        self.timestep = None
        self.episode = None

        self.model = self.initialize_model()
        if self.batched_observe:
            assert self.batching_capacity is not None
            self.observe_terminal = [list() for _ in range(self.model.num_parallel)]
            self.observe_reward = [list() for _ in range(self.model.num_parallel)]
        self.reset()

    def __str__(self):
        return str(self.__class__.__name__)

    def close(self):
        self.model.close()

    def initialize_model(self):
        """
        Creates and returns the model (including a local replica in case of distributed learning) for this agent
        based on specifications given by user. This method needs to be implemented by the different agent subclasses.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the agent to its initial state (e.g. on experiment start). Updates the Model's internal episode and
        time step counter, internal states, and resets preprocessors.
        """
        self.episode, self.timestep, self.next_internals = self.model.reset()
        self.current_internals = self.next_internals

    def act(self, states, deterministic=False, independent=False, fetch_tensors=None, buffered=True, index=0):
        """
        Return action(s) for given state(s). States preprocessing and exploration are applied if
        configured accordingly.

        Args:
            states (any): One state (usually a value tuple) or dict of states if multiple states are expected.
            deterministic (bool): If true, no exploration and sampling is applied.
            independent (bool): If true, action is not followed by observe (and hence not included
                in updates).
            fetch_tensors (list): Optional String of named tensors to fetch
            buffered (bool): If true (default), states and internals are not returned but buffered
                with observes. Must be false for multi-threaded mode as we need atomic inserts.
        Returns:
            Scalar value of the action or dict of multiple actions the agent wants to execute.
            (fetched_tensors) Optional dict() with named tensors fetched
        """
        self.current_internals = self.next_internals

        if self.unique_state:
            self.current_states = dict(state=np.asarray(states))
        else:
            self.current_states = {name: np.asarray(states[name]) for name in sorted(states)}

        if fetch_tensors is not None:
            # Retrieve action
            self.current_actions, self.next_internals, self.timestep, self.fetched_tensors = self.model.act(
                states=self.current_states,
                internals=self.current_internals,
                deterministic=deterministic,
                independent=independent,
                fetch_tensors=fetch_tensors,
                index=index
            )

            if self.unique_action:
                return self.current_actions['action'], self.fetched_tensors
            else:
                return self.current_actions, self.fetched_tensors

        # Retrieve action.
        self.current_actions, self.next_internals, self.timestep = self.model.act(
            states=self.current_states,
            internals=self.current_internals,
            deterministic=deterministic,
            independent=independent,
            index=index
        )

        # Buffered mode only works single-threaded because buffer inserts
        # by multiple threads are non-atomic and can cause race conditions.
        if buffered:
            if self.unique_action:
                return self.current_actions['action']
            else:
                return self.current_actions
        else:
            if self.unique_action:
                return self.current_actions['action'], self.current_states, self.current_internals
            else:
                return self.current_actions, self.current_states, self.current_internals

    def observe(self, terminal, reward, index=0):
        """
        Observe experience from the environment to learn from. Optionally pre-processes rewards
        Child classes should call super to get the processed reward
        EX: terminal, reward = super()...

        Args:
            terminal (bool): boolean indicating if the episode terminated after the observation.
            reward (float): scalar reward that resulted from executing the action.
        """
        self.current_terminal = terminal
        self.current_reward = reward

        if self.batched_observe:
            # Batched observe for better performance with Python.
            self.observe_terminal[index].append(self.current_terminal)
            self.observe_reward[index].append(self.current_reward)

            if self.current_terminal or len(self.observe_terminal[index]) >= self.batching_capacity:
                self.episode = self.model.observe(
                    terminal=self.observe_terminal[index],
                    reward=self.observe_reward[index],
                    index=index
                )
                self.observe_terminal[index] = list()
                self.observe_reward[index] = list()

        else:
            self.episode = self.model.observe(
                terminal=self.current_terminal,
                reward=self.current_reward
            )

    def atomic_observe(self, states, actions, internals, reward, terminal):
        """
        Utility method for unbuffered observing where each tuple is inserted into TensorFlow via
        a single session call, thus avoiding race conditions in multi-threaded mode.

        Observe full experience  tuplefrom the environment to learn from. Optionally pre-processes rewards
        Child classes should call super to get the processed reward
        EX: terminal, reward = super()...

        Args:
            states (any): One state (usually a value tuple) or dict of states if multiple states are expected.
            actions (any): One action (usually a value tuple) or dict of states if multiple actions are expected.
            internals (any): Internal list.
            terminal (bool): boolean indicating if the episode terminated after the observation.
            reward (float): scalar reward that resulted from executing the action.
        """
        # TODO probably unnecessary here.
        self.current_terminal = terminal
        self.current_reward = reward
        # print('action = {}'.format(actions))
        if self.unique_state:
            states = dict(state=states)
        if self.unique_action:
            actions = dict(action=actions)

        self.episode = self.model.atomic_observe(
            states=states,
            actions=actions,
            internals=internals,
            terminal=self.current_terminal,
            reward=self.current_reward
        )

    def should_stop(self):
        return self.model.monitored_session.should_stop()

    def last_observation(self):
        return dict(
            states=self.current_states,
            internals=self.current_internals,
            actions=self.current_actions,
            terminal=self.current_terminal,
            reward=self.current_reward
        )

    def save_model(self, directory=None, append_timestep=True):
        """
        Save TensorFlow model. If no checkpoint directory is given, the model's default saver
        directory is used. Optionally appends current timestep to prevent overwriting previous
        checkpoint files. Turn off to be able to load model from the same given path argument as
        given here.

        Args:
            directory (str): Optional checkpoint directory.
            append_timestep (bool):  Appends the current timestep to the checkpoint file if true.
                If this is set to True, the load path must include the checkpoint timestep suffix.
                For example, if stored to models/ and set to true, the exported file will be of the
                form models/model.ckpt-X where X is the last timestep saved. The load path must
                precisely match this file name. If this option is turned off, the checkpoint will
                always overwrite the file specified in path and the model can always be loaded under
                this path.

        Returns:
            Checkpoint path were the model was saved.
        """
        return self.model.save(directory=directory, append_timestep=append_timestep)

    def restore_model(self, directory=None, file=None):
        """
        Restore TensorFlow model. If no checkpoint file is given, the latest checkpoint is
        restored. If no checkpoint directory is given, the model's default saver directory is
        used (unless file specifies the entire path).

        Args:
            directory: Optional checkpoint directory.
            file: Optional checkpoint file, or path if directory not given.
        """
        self.model.restore(directory=directory, file=file)

    @staticmethod
    def from_spec(spec, kwargs):
        """
        Creates an agent from a specification dict.
        """
        agent = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.agents.agents,
            kwargs=kwargs
        )
        assert isinstance(agent, Agent)
        return agent
