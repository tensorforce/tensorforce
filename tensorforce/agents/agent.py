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

import json
import numpy as np
import os

from tensorforce import util, TensorforceError
import tensorforce.agents


class Agent(object):
    """
    Base class for Tensorforce agents.
    """

    @staticmethod
    def from_spec(spec, **kwargs):
        """
        Creates an agent from a specification.
        """
        if isinstance(spec, str):
            assert os.path.isfile(spec)
            # JSON file specification
            with open(spec, 'r') as fp:
                spec = json.load(fp=fp)

        for key, arg in kwargs.items():
            if key not in spec:
                spec[key] = arg

        agent = spec.pop('type')
        agent = tensorforce.agents.agents[agent](**spec)
        assert isinstance(agent, Agent)

        return agent

    def __init__(
        self, states, actions, parallel_interactions=1, buffer_observe=1000
    ):
        """
        Agent constructor.

        Args:
            states (specification): States specification, arbitrarily nested dictionary of state
                descriptions with the following attributes:
                - type ('bool' | 'int' | 'float'): state data type (default: 'float').
                - shape (int | iter[int]): state shape (required).
                - num_states (int > 0): number of discrete state values (required for type 'int').
                - min_value/max_value (float): minimum/maximum state value (optional for type
                'float').
            actions (specification): Actions specification, arbitrarily nested dictionary of action
                descriptions with the following attributes:
                - type ('bool' | 'int' | 'float'): action data type (required).
                - shape (int > 0 | iter[int > 0]): action shape (default: []).
                - num_actions (int > 0): number of discrete action values (required for type
                'int').
                - min_value/max_value (float): minimum/maximum action value (optional for type
                'float').
            parallel_interactions (int > 0): Maximum number of parallel interactions to support,
                for instance, to enable multiple parallel episodes, environments or (centrally
                controlled) agents within an environment.
            buffer_observe (int > 0): Maximum number of timesteps within an episode to buffer
                before executing internal observe operations, to reduce calls to TensorFlow for
                improved performance.
        """

        # States/actions specification
        self.states_spec = util.valid_values_spec(
            values_spec=states, value_type='state', return_normalized=True
        )
        self.actions_spec = util.valid_values_spec(
            values_spec=actions, value_type='action', return_normalized=True
        )

        # Check for name overlap
        for name in self.states_spec:
            if name in self.actions_spec:
                TensorforceError.collision(
                    name='name', value=name, group1='states', group2='actions'
                )

        # Parallel episodes
        if isinstance(parallel_interactions, int):
            if parallel_interactions <= 0:
                raise TensorforceError.value(
                    name='parallel_interactions', value=parallel_interactions
                )
            self.parallel_interactions = parallel_interactions
        else:
            raise TensorforceError.type(name='parallel_interactions', value=parallel_interactions)

        # Buffer observe
        if isinstance(buffer_observe, bool):
            # if update_mode['unit'] == 'episodes':
            #     self.buffer_observe = 1000 if buffer_observe else 1
            # else:
            #     self.buffer_observe = update_mode['batch_size']
            self.buffer_observe = 1000 if buffer_observe else 1
        elif isinstance(buffer_observe, int):
            if buffer_observe <= 0:
                raise TensorforceError.value(name='buffer_observe', value=buffer_observe)
            self.buffer_observe = buffer_observe
        else:
            raise TensorforceError.type(name='buffer_observe', value=buffer_observe)

        # Parallel terminal/reward buffers
        self.terminal_buffers = np.ndarray(
            shape=(self.parallel_interactions, self.buffer_observe),
            dtype=util.np_dtype(dtype='bool')
        )
        self.reward_buffers = np.ndarray(
            shape=(self.parallel_interactions, self.buffer_observe),
            dtype=util.np_dtype(dtype='float')
        )

        # Parallel buffer indices
        self.buffer_indices = np.zeros(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='int')
        )

        self.timestep = 0
        self.episode = 0

    def initialize(self):
        if not hasattr(self, 'model'):
            raise TensorforceError.missing(name='Agent', value='model')

        # Setup Model (create and build graph (local and global if distributed), server, session, etc..).
        if not self.model.is_initialized:
            self.model.setup()  # should be self.model.initialize()

    def reset(self):
        self.buffer_indices = np.zeros(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='int')
        )
        self.model.reset()

    def close(self):
        self.model.close()

    def initialize_model(self):
        """
        Creates and returns the model (including a local replica in case of distributed learning) for this agent
        based on specifications given by user. This method needs to be implemented by the different agent subclasses.
        """
        raise NotImplementedError

    def act(
        self, states, parallel=0, deterministic=False, independent=False, query=None, **kwargs
    ):
        """
        Return action(s) for given state(s). States preprocessing and exploration are applied if
        configured accordingly.

        Args:
            states (any): One state (usually a value tuple) or dict of states if multiple states are expected.
            deterministic (bool): If true, no exploration and sampling is applied.
            independent (bool): If true, action is not followed by observe (and hence not included
                in updates).
            fetch_tensors (list): Optional String of named tensors to fetch
        Returns:
            Scalar value of the action or dict of multiple actions the agent wants to execute.
            (fetched_tensors) Optional dict() with named tensors fetched
        """
        # self.current_internals = self.next_internals

        # Normalize states dictionary
        states = util.normalize_values(
            value_type='state', values=states, values_spec=self.states_spec
        )

        # Batch states
        states = util.fmap(function=(lambda x: [x]), xs=states)

        # Model.act()
        if query is None:
            actions, self.timestep = self.model.act(
                states=states, parallel=parallel, deterministic=deterministic,
                independent=independent, **kwargs
            )

        else:
            actions, self.timestep, query = self.model.act(
                states=states, parallel=parallel, deterministic=deterministic,
                independent=independent, query=query, **kwargs
            )

        # Unbatch actions
        actions = util.fmap(function=(lambda x: x[0]), xs=actions)

        # Reverse normalized actions dictionary
        actions = util.unpack_values(
            value_type='action', values=actions, values_spec=self.actions_spec
        )

        # if independent, return processed state as well?

        if query is None:
            return actions
        else:
            return actions, query

    def observe(self, terminal, reward, parallel=0, query=None, **kwargs):
        """
        Observe experience from the environment to learn from. Optionally pre-processes rewards
        Child classes should call super to get the processed reward
        EX: terminal, reward = super()...

        Args:
            terminal (bool): boolean indicating if the episode terminated after the observation.
            reward (float): scalar reward that resulted from executing the action.
        """

        # Update terminal/reward buffer
        index = self.buffer_indices[parallel]
        self.terminal_buffers[parallel, index] = terminal
        self.reward_buffers[parallel, index] = reward
        index += 1

        if terminal or index == self.buffer_observe or query is not None:
            # Model.observe()
            if query is None:
                self.episode = self.model.observe(
                    terminal=self.terminal_buffers[parallel, :index],
                    reward=self.reward_buffers[parallel, :index], parallel=parallel, **kwargs
                )

            else:
                self.episode, query = self.model.observe(
                    terminal=self.terminal_buffers[parallel, :index],
                    reward=self.reward_buffers[parallel, :index], parallel=parallel, query=query,
                    **kwargs
                )

            # Reset buffer index
            self.buffer_indices[parallel] = 0

        else:
            # Increment buffer index
            self.buffer_indices[parallel] = index

        if query is not None:
            return query

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

    def save_model(self, directory=None, filename=None, append_timestep=True):
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
        # Empty buffers before saving
        for parallel in range(self.parallel_interactions):
            index = self.buffer_indices[parallel]
            if index > 0:
                self.episode = self.model.observe(
                    terminal=self.terminal_buffers[parallel, :index],
                    reward=self.reward_buffers[parallel, :index], parallel=parallel
                )
                self.buffer_indices[parallel] = 0

        return self.model.save(
            directory=directory, filename=filename, append_timestep=append_timestep
        )

    def restore_model(self, directory=None, filename=None):
        """
        Restore TensorFlow model. If no checkpoint file is given, the latest checkpoint is
        restored. If no checkpoint directory is given, the model's default saver directory is
        used (unless file specifies the entire path).

        Args:
            directory: Optional checkpoint directory.
            file: Optional checkpoint file, or path if directory not given.
        """
        if not hasattr(self, 'model'):
            raise TensorforceError.missing(name='Agent', value='model')

        if not self.model.is_initialized:
            self.model.setup()

        self.model.restore(directory=directory, filename=filename)
