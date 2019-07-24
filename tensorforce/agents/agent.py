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

from collections import OrderedDict
import importlib
import json
import logging
import os
import random

import numpy as np
import tensorflow as tf

from tensorforce import util, TensorforceError
import tensorforce.agents


class Agent(object):
    """
    Tensorforce agent interface.
    """

    @staticmethod
    def create(agent=None, environment=None, **kwargs):
        """
        Creates an agent from a specification.

        Args:
            agent (specification): JSON file, specification key, configuration dictionary,
                library module, or `Agent` subclass
                (<span style="color:#00C000"><b>default</b></span>: Policy agent).
            environment (Environment): Environment which the agent is supposed to be trained on,
                environment-related arguments like state/action space specifications will be
                extract if given.
            kwargs: Additional arguments.
        """
        if agent is None:
            agent = 'default'

        if isinstance(agent, Agent):
            # TODO: asserts???????
            return agent

        elif isinstance(agent, dict):
            # Dictionary specification
            kwargs.update(agent)
            agent = kwargs.pop('agent', kwargs.pop('type', 'default'))

            return Agent.create(agent=agent, environment=environment, **kwargs)

        elif isinstance(agent, str):
            if os.path.isfile(agent):
                # JSON file specification
                with open(agent, 'r') as fp:
                    agent = json.load(fp=fp)

                kwargs.update(agent)
                agent = kwargs.pop('agent', kwargs.pop('type', 'default'))

                return Agent.create(agent=agent, environment=environment, **kwargs)

            elif '.' in agent:
                # Library specification
                library_name, module_name = agent.rsplit('.', 1)
                library = importlib.import_module(name=library_name)
                agent = getattr(library, module_name)

                if 'states' not in kwargs:  # TODO: otherwise exception
                    kwargs['states'] = environment.states()
                if 'actions' not in kwargs:
                    kwargs['actions'] = environment.actions()
                if 'max_episode_timesteps' not in kwargs:
                    if environment.max_episode_timesteps() is not None:
                        kwargs['max_episode_timesteps'] = environment.max_episode_timesteps()

                agent = agent(**kwargs)
                assert isinstance(agent, Agent)

                return agent

            else:
                # Keyword specification
                if 'states' not in kwargs:  # TODO: otherwise exception
                    kwargs['states'] = environment.states()
                if 'actions' not in kwargs:
                    kwargs['actions'] = environment.actions()
                if 'max_episode_timesteps' not in kwargs:
                    if environment.max_episode_timesteps() is not None:
                        kwargs['max_episode_timesteps'] = environment.max_episode_timesteps()

                agent = tensorforce.agents.agents[agent](**kwargs)
                assert isinstance(agent, Agent)

                return agent

        else:
            assert False

    def __init__(
        # Environment
        self, states, actions, max_episode_timesteps=None,
        # TensorFlow etc
        parallel_interactions=1, buffer_observe=True, seed=None
    ):
        if seed is not None:
            assert isinstance(seed, int)
            random.seed(n=seed)
            np.random.seed(seed=seed)
            tf.random.set_random_seed(seed=seed)

        # States/actions specification
        self.states_spec = util.valid_values_spec(
            values_spec=states, value_type='state', return_normalized=True
        )
        self.actions_spec = util.valid_values_spec(
            values_spec=actions, value_type='action', return_normalized=True
        )
        self.max_episode_timesteps = max_episode_timesteps

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
            if not buffer_observe and self.parallel_interactions > 1:
                raise TensorforceError.unexpected()
            if self.max_episode_timesteps is None and self.parallel_interactions > 1:
                raise TensorforceError.unexpected()
            if not buffer_observe:
                self.buffer_observe = 1
            elif self.max_episode_timesteps is None:
                self.buffer_observe = 100
            else:
                self.buffer_observe = self.max_episode_timesteps
        elif isinstance(buffer_observe, int):
            if buffer_observe <= 0:
                raise TensorforceError.value(name='buffer_observe', value=buffer_observe)
            if self.parallel_interactions > 1:
                raise TensorforceError.unexpected()
            if self.max_episode_timesteps is None:
                self.buffer_observe = buffer_observe
            else:
                self.buffer_observe = min(buffer_observe, self.max_episode_timesteps)
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

    def __str__(self):
        return self.__class__.__name__

    def initialize(self):
        """
        Initializes the agent.
        """
        if not hasattr(self, 'model'):
            raise TensorforceError.missing(name='Agent', value='model')

        # Setup Model (create and build graph (local and global if distributed), server, session, etc..).
        self.model.initialize()
        self.reset()

    def close(self):
        """
        Closes the agent.
        """
        self.model.close()

    def reset(self):
        """
        Resets the agent to start a new episode.
        """
        self.buffer_indices = np.zeros(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='int')
        )
        self.timestep, self.episode = self.model.reset()

    def act(
        self, states, parallel=0, deterministic=False, independent=False, evaluation=False,
        query=None, **kwargs
    ):
        """
        Returns action(s) for the given state(s), needs to be followed by `observe(...)` unless
        `independent` is true.

        Args:
            states (dict[state]): Dictionary containing state(s) to be acted on
                (<span style="color:#C00000"><b>required</b></span>).
            parallel (int): Parallel execution index
                (<span style="color:#00C000"><b>default</b></span>: 0).
            deterministic (bool): Whether to apply exploration and sampling
                (<span style="color:#00C000"><b>default</b></span>: false).
            independent (bool): Whether action is not remembered, and this call is thus not
                followed by observe
                (<span style="color:#00C000"><b>default</b></span>: false).
            evaluation (bool): Whether the agent is currently evaluated, implies and overwrites
                deterministic and independent
                (<span style="color:#00C000"><b>default</b></span>: false).
            query (list[str]): Names of tensors to retrieve
                (<span style="color:#00C000"><b>default</b></span>: none).
            kwargs: Additional input values, for instance, for dynamic hyperparameters.

        Returns:
            (dict[action], plus optional list[str]): Dictionary containing action(s), plus queried
            tensor values if requested.
        """
        # self.current_internals = self.next_internals
        if evaluation:
            if deterministic or independent:
                raise TensorforceError.unexpected()
            deterministic = independent = True

        # Auxiliaries
        auxiliaries = OrderedDict()
        if isinstance(states, dict):
            for name, spec in self.actions_spec.items():
                if spec['type'] == 'int' and name + '_mask' in states:
                    auxiliaries[name + '_mask'] = states.pop(name + '_mask')

        # Normalize states dictionary
        states = util.normalize_values(
            value_type='state', values=states, values_spec=self.states_spec
        )

        # Batch states
        states = util.fmap(function=(lambda x: [x]), xs=states)
        auxiliaries = util.fmap(function=(lambda x: [x]), xs=auxiliaries)

        # Model.act()
        if query is None:
            actions, self.timestep = self.model.act(
                states=states, auxiliaries=auxiliaries, parallel=parallel,
                deterministic=deterministic, independent=independent, **kwargs
            )

        else:
            actions, self.timestep, queried = self.model.act(
                states=states, auxiliaries=auxiliaries, parallel=parallel,
                deterministic=deterministic, independent=independent, query=query, **kwargs
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
            return actions, queried

    def observe(self, reward, terminal=False, parallel=0, query=None, **kwargs):
        """
        Observes reward and whether a terminal state is reached, needs to be preceded by
        `act(...)`.

        Args:
            reward (float): Reward
                (<span style="color:#C00000"><b>required</b></span>).
            terminal (bool): Whether a terminal state is reached
                (<span style="color:#00C000"><b>default</b></span>: false).
            parallel (int): Parallel execution index
                (<span style="color:#00C000"><b>default</b></span>: 0).
            query (list[str]): Names of tensors to retrieve
                (<span style="color:#00C000"><b>default</b></span>: none).
            kwargs: Additional input values, for instance, for dynamic hyperparameters.

        Returns:
            (optional list[str]): Queried tensor values if requested.
        """
        if query is not None and self.parallel_interactions > 1:
            raise TensorforceError.unexpected()

        # Update terminal/reward buffer
        index = self.buffer_indices[parallel]
        self.terminal_buffers[parallel, index] = terminal
        self.reward_buffers[parallel, index] = reward
        index += 1

        if self.max_episode_timesteps is not None and index > self.max_episode_timesteps:
            raise TensorforceError.unexpected()

        if terminal or index == self.buffer_observe or query is not None:
            # Model.observe()
            if query is None:
                self.episode = self.model.observe(
                    terminal=self.terminal_buffers[parallel, :index],
                    reward=self.reward_buffers[parallel, :index], parallel=parallel, **kwargs
                )

            else:
                self.episode, queried = self.model.observe(
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
            return queried

    def save(self, directory=None, filename=None, append_timestep=True):
        """
        Saves the current state of the agent.

        Args:
            directory (str): Checkpoint directory
                (<span style="color:#00C000"><b>default</b></span>: directory specified for
                TensorFlow saver).
            filename (str): Checkpoint filename
                (<span style="color:#00C000"><b>default</b></span>: filename specified for
                TensorFlow saver).
            append_timestep: Whether to append the current timestep to the checkpoint file
                (<span style="color:#00C000"><b>default</b></span>: true).

        Returns:
            str: Checkpoint path.
        """
        # TODO: Messes with required parallel disentangling, better to remove unfinished episodes
        # from memory, but currently entire episode buffered anyway...
        # # Empty buffers before saving
        # for parallel in range(self.parallel_interactions):
        #     index = self.buffer_indices[parallel]
        #     if index > 0:
        #         # if self.parallel_interactions > 1:
        #         #     raise TensorforceError.unexpected()
        #         self.episode = self.model.observe(
        #             terminal=self.terminal_buffers[parallel, :index],
        #             reward=self.reward_buffers[parallel, :index], parallel=parallel
        #         )
        #         self.buffer_indices[parallel] = 0

        return self.model.save(
            directory=directory, filename=filename, append_timestep=append_timestep
        )

    def restore(self, directory=None, filename=None):
        """
        Restores the agent.

        Args:
            directory (str): Checkpoint directory
                (<span style="color:#00C000"><b>default</b></span>: directory specified for
                TensorFlow saver).
            filename (str): Checkpoint filename
                (<span style="color:#00C000"><b>default</b></span>: latest checkpoint in
                directory).
        """
        if not hasattr(self, 'model'):
            raise TensorforceError.missing(name='Agent', value='model')

        if not self.model.is_initialized:
            self.model.initialize()

        self.timestep, self.episode = self.model.restore(directory=directory, filename=filename)

    def get_output_tensors(self, function):
        """
        Returns the names of output tensors for the given function.

        Args:
            function (str): Function name
                (<span style="color:#C00000"><b>required</b></span>).

        Returns:
            list[str]: Names of output tensors.
        """
        if function in self.model.output_tensors:
            return self.model.output_tensors[function]
        else:
            raise TensorforceError.unexpected()

    def get_query_tensors(self, function):
        """
        Returns the names of queryable tensors for the given function.

        Args:
            function (str): Function name
                (<span style="color:#C00000"><b>required</b></span>).

        Returns:
            list[str]: Names of queryable tensors.
        """
        if function in self.model.query_tensors:
            return self.model.query_tensors[function]
        else:
            raise TensorforceError.unexpected()

    def get_available_summaries(self):
        """
        Returns the summary labels provided by the agent.

        Returns:
            list[str]: Available summary labels.
        """
        return self.model.get_available_summaries()

    def should_stop(self):
        return self.model.monitored_session.should_stop()
