# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

import importlib
import json
import os
import random
import time
from collections import OrderedDict

import numpy as np

import tensorforce.agents
from tensorforce import util, TensorforceError
from tensorforce.core.utils.json_encoder import NumpyJSONEncoder


class Agent(object):
    """
    Tensorforce agent interface.
    """

    @staticmethod
    def create(agent='tensorforce', environment=None, **kwargs):
        """
        Creates an agent from a specification.

        Args:
            agent (specification | Agent class/object): JSON file, specification key, configuration
                dictionary, library module, or `Agent` class/object
                (<span style="color:#00C000"><b>default</b></span>: Policy agent).
            environment (Environment object): Environment which the agent is supposed to be trained
                on, environment-related arguments like state/action space specifications and
                maximum episode length will be extract if given
                (<span style="color:#00C000"><b>recommended</b></span>).
            kwargs: Additional arguments.
        """
        if isinstance(agent, Agent):
            if environment is not None:
                assert util.deep_equal(xs=agent.spec['states'], ys=environment.states())
                assert util.deep_equal(xs=agent.spec['actions'], ys=environment.actions())
                assert environment.max_episode_timesteps() is None or \
                    agent.spec['max_episode_timesteps'] >= environment.max_episode_timesteps()

            for key, value in kwargs.items():
                assert agent.spec[key] == value

            if agent.is_initialized:
                agent.reset()
            else:
                agent.initialize()

            return agent

        elif isinstance(agent, type) and issubclass(agent, Agent):
            if environment is not None:
                if 'states' in kwargs:
                    assert util.deep_equal(xs=kwargs['states'], ys=environment.states())
                else:
                    kwargs['states'] = environment.states()
                if 'actions' in kwargs:
                    assert util.deep_equal(xs=kwargs['actions'], ys=environment.actions())
                else:
                    kwargs['actions'] = environment.actions()
                if environment.max_episode_timesteps() is None:
                    pass
                elif 'max_episode_timesteps' in kwargs:
                    assert kwargs['max_episode_timesteps'] >= environment.max_episode_timesteps()
                else:
                    kwargs['max_episode_timesteps'] = environment.max_episode_timesteps()

            agent = agent(**kwargs)
            assert isinstance(agent, Agent)
            return Agent.create(agent=agent, environment=environment)

        elif isinstance(agent, dict):
            # Dictionary specification
            util.deep_disjoint_update(target=kwargs, source=agent)
            agent = kwargs.pop('agent', kwargs.pop('type', 'tensorforce'))

            return Agent.create(agent=agent, environment=environment, **kwargs)

        elif isinstance(agent, str):
            if os.path.isfile(agent):
                # JSON file specification
                with open(agent, 'r') as fp:
                    agent = json.load(fp=fp)
                return Agent.create(agent=agent, environment=environment, **kwargs)

            elif '.' in agent:
                # Library specification
                library_name, module_name = agent.rsplit('.', 1)
                library = importlib.import_module(name=library_name)
                agent = getattr(library, module_name)
                return Agent.create(agent=agent, environment=environment, **kwargs)

            else:
                # Keyword specification
                agent = tensorforce.agents.agents[agent]
                return Agent.create(agent=agent, environment=environment, **kwargs)

        else:
            raise TensorforceError.type(name='Agent.create', argument='agent', dtype=type(agent))

    @staticmethod
    def load(directory, filename=None, environment=None, **kwargs):
        """
        Restores an agent from a specification directory/file.

        Args:
            directory (str): Agent directory
                (<span style="color:#C00000"><b>required</b></span>).
            filename (str): Agent filename
                (<span style="color:#00C000"><b>default</b></span>: "agent").
            environment (Environment object): Environment which the agent is supposed to be trained
                on, environment-related arguments like state/action space specifications and
                maximum episode length will be extract if given
                (<span style="color:#00C000"><b>recommended</b></span>).
            kwargs: Additional arguments.
        """
        if filename is None:
            agent = os.path.join(directory, 'agent.json')
        else:
            agent = os.path.join(directory, filename + '.json')

        assert os.path.isfile(agent)
        with open(agent, 'r') as fp:
            agent = json.load(fp=fp)

        # Overwrite values
        if environment is not None and environment.max_episode_timesteps() is not None:
            if 'max_episode_timesteps' in kwargs:
                assert kwargs['max_episode_timesteps'] >= environment.max_episode_timesteps()
                agent['max_episode_timesteps'] = kwargs['max_episode_timesteps']
            else:
                agent['max_episode_timesteps'] = environment.max_episode_timesteps()
        if 'parallel_interactions' in kwargs:
            agent['parallel_interactions'] = kwargs['parallel_interactions']

        agent = Agent.create(agent=agent, environment=environment, **kwargs)
        agent.restore(directory=directory, filename=filename)
        return agent

    def __init__(
        # Environment
        self, states, actions, max_episode_timesteps=None,
        # TensorFlow etc
        parallel_interactions=1, buffer_observe=True, seed=None, recorder=None
    ):
        assert hasattr(self, 'spec')

        if seed is not None:
            assert isinstance(seed, int)
            random.seed(a=seed)
            np.random.seed(seed=seed)

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
                    name='agent', argument='parallel_interactions', value=parallel_interactions,
                    hint='<= 0'
                )
            self.parallel_interactions = parallel_interactions
        else:
            raise TensorforceError.type(
                name='agent', argument='parallel_interactions', dtype=type(parallel_interactions)
            )

        # Buffer observe
        if isinstance(buffer_observe, bool):
            if self.parallel_interactions > 1:
                if not buffer_observe:
                    raise TensorforceError.required(
                        name='agent', argument='buffer_observe',
                        condition='parallel_interactions > 1'
                    )
                elif self.max_episode_timesteps is None:
                    raise TensorforceError.required(
                        name='agent', argument='max_episode_timesteps',
                        condition='parallel_interactions > 1'
                    )
            if not buffer_observe:
                self.buffer_observe = 1
            elif self.max_episode_timesteps is None:
                self.buffer_observe = 100
            else:
                self.buffer_observe = self.max_episode_timesteps
        elif isinstance(buffer_observe, int):
            if buffer_observe <= 0:
                raise TensorforceError.value(
                    name='agent', argument='buffer_observe', value=buffer_observe, hint='<= 0'
                )
            if self.parallel_interactions > 1:
                raise TensorforceError.value(
                    name='agent', argument='buffer_observe', value=buffer_observe,
                    condition='parallel_interactions > 1'
                )
            if self.max_episode_timesteps is None:
                self.buffer_observe = buffer_observe
            else:
                self.buffer_observe = min(buffer_observe, self.max_episode_timesteps)
        else:
            raise TensorforceError.type(
                name='agent', argument='buffer_observe', dtype=type(buffer_observe)
            )

        # Recorder
        if recorder is None:
            pass
        elif not all(key in ('directory', 'frequency', 'max-traces', 'start') for key in recorder):
            raise TensorforceError.value(
                name='agent', argument='recorder', value=list(recorder),
                hint='not from {directory,frequency,max-traces,start}'
            )
        self.recorder_spec = recorder if recorder is None else dict(recorder)

        self.is_initialized = False

    def __str__(self):
        return self.__class__.__name__

    def initialize(self):
        """
        Initializes the agent, usually done as part of Agent.create/load.
        """
        if self.is_initialized:
            raise TensorforceError(
                message="Agent is already initialized, possibly as part of Agent.create()."
            )

        self.is_initialized = True

        # Parallel terminal/reward buffers
        self.terminal_buffers = np.ndarray(
            shape=(self.parallel_interactions, self.buffer_observe),
            dtype=util.np_dtype(dtype='long')
        )
        self.reward_buffers = np.ndarray(
            shape=(self.parallel_interactions, self.buffer_observe),
            dtype=util.np_dtype(dtype='float')
        )

        # Recorder buffers if required
        if self.recorder_spec is not None:
            self.states_buffers = OrderedDict()
            self.actions_buffers = OrderedDict()
            for name, spec in self.states_spec.items():
                shape = (self.parallel_interactions, self.buffer_observe) + spec['shape']
                self.states_buffers[name] = np.ndarray(
                    shape=shape, dtype=util.np_dtype(dtype=spec['type'])
                )
            for name, spec in self.actions_spec.items():
                shape = (self.parallel_interactions, self.buffer_observe) + spec['shape']
                self.actions_buffers[name] = np.ndarray(
                    shape=shape, dtype=util.np_dtype(dtype=spec['type'])
                )
                if spec['type'] == 'int':
                    shape = (self.parallel_interactions, self.buffer_observe) + spec['shape'] + \
                        (spec['num_values'],)
                    self.states_buffers[name + '_mask'] = np.ndarray(
                        shape=shape, dtype=util.np_dtype(dtype='bool')
                    )

            self.num_episodes = 0
            self.record_states = OrderedDict(((name, list()) for name in self.states_spec))
            self.record_actions = OrderedDict(((name, list()) for name in self.actions_spec))
            for name, spec in self.actions_spec.items():
                if spec['type'] == 'int':
                    self.record_states[name + '_mask'] = list()
            self.record_terminal = list()
            self.record_reward = list()

        # Parallel buffer indices
        self.buffer_indices = np.zeros(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='int')
        )

        self.timesteps = 0
        self.episodes = 0
        self.updates = 0

        # Setup Model
        if not hasattr(self, 'model'):
            raise TensorforceError(message="Missing agent attribute model.")

        self.model.initialize()
        if self.model.saver_directory is not None:
            file = os.path.join(self.model.saver_directory, self.model.saver_filename + '.json')
            with open(file, 'w') as fp:
                json.dump(obj=self.spec, fp=fp, cls=NumpyJSONEncoder)

        self.reset()

    def close(self):
        """
        Closes the agent.
        """
        self.model.close()
        self.model = None

    def reset(self, independent=False, evaluation=False):
        """
        Resets all agent buffers, or only terminates and resets an episode in
            independent/evaluation mode if corresponding argument is set.

        Args:
            independent (bool): Whether to terminate an episode in independent mode
                (<span style="color:#00C000"><b>default</b></span>: false).
            evaluation (bool): Whether to terminate an episode in evaluation mode, implies
                independent
                (<span style="color:#00C000"><b>default</b></span>: false).
        """
        if evaluation:
            if independent:
                raise TensorforceError(
                    message="Agent.reset argument independent is implied by and thus should not "
                            "be used together with argument evaluation."
                )
            independent = True

        if not independent:
            self.buffer_indices = np.zeros(
                shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='int')
            )

        self.timesteps, self.episodes, self.updates = self.model.reset(independent=independent)

    def act(
        self, states, parallel=0, deterministic=False, independent=False, evaluation=False,
        query=None, **kwargs
    ):
        """
        Returns action(s) for the given state(s), needs to be followed by `observe(...)`, or
        terminated by `reset()` in case of `independent`/`evaluation`.

        Args:
            states (dict[state] | iter[dict[state]]): Dictionary containing state(s) to be acted on
                (<span style="color:#C00000"><b>required</b></span>).
            parallel (int | iter[int]): Parallel execution index
                (<span style="color:#00C000"><b>default</b></span>: 0).
            deterministic (bool): Whether to apply exploration and sampling
                (<span style="color:#00C000"><b>default</b></span>: false).
            independent (bool): Whether action is not remembered, and this call is thus not
                followed by observe
                (<span style="color:#00C000"><b>default</b></span>: false).
            evaluation (bool): Whether the agent is currently evaluated, implies deterministic and
                independent
                (<span style="color:#00C000"><b>default</b></span>: false).
            query (list[str]): Names of tensors to retrieve
                (<span style="color:#00C000"><b>default</b></span>: none).
            kwargs: Additional input values, for instance, for dynamic hyperparameters.

        Returns:
            (dict[action] | iter[dict[action]], plus optional list[str]): Dictionary containing
            action(s), plus queried tensor values if requested.
        """
        assert util.reduce_all(predicate=util.not_nan_inf, xs=states)

        if evaluation:
            if deterministic or independent:
                raise TensorforceError(
                    message="Agent.observe argument deterministic/independent are implied by and "
                            "thus should not be used together with argument evaluation."
                )
            deterministic = independent = True

        # Batch states
        batched = (not isinstance(parallel, int))
        if batched:
            if len(parallel) == 0:
                raise TensorforceError.value(
                    name='agent.act', argument='parallel', value=parallel, hint='zero-length'
                )
            parallel = np.asarray(list(parallel))
            if isinstance(states[0], dict):
                states = OrderedDict((
                    (name, np.asarray([states[n][name] for n in range(len(parallel))]))
                    for name in states[0]
                ))
            else:
                states = np.asarray(states)
        else:
            parallel = np.asarray([parallel])
            states = util.fmap(
                function=(lambda x: np.asarray([x])), xs=states,
                depth=int(isinstance(states, dict))
            )

        # Auxiliaries
        auxiliaries = OrderedDict()
        if isinstance(states, dict):
            states = dict(states)
            for name, spec in self.actions_spec.items():
                if spec['type'] == 'int' and name + '_mask' in states:
                    auxiliaries[name + '_mask'] = states.pop(name + '_mask')

        # Normalize states dictionary
        states = util.normalize_values(
            value_type='state', values=states, values_spec=self.states_spec
        )

        # Model.act()
        if query is None:
            actions, self.timesteps = self.model.act(
                states=states, auxiliaries=auxiliaries, parallel=parallel,
                deterministic=deterministic, independent=independent, **kwargs
            )

        else:
            actions, self.timesteps, queried = self.model.act(
                states=states, auxiliaries=auxiliaries, parallel=parallel,
                deterministic=deterministic, independent=independent, query=query, **kwargs
            )

        if self.recorder_spec is not None and not independent and \
                self.episodes >= self.recorder_spec.get('start', 0):
            for n in range(len(parallel)):
                index = self.buffer_indices[parallel[n]]
                for name in self.states_spec:
                    self.states_buffers[name][parallel[n], index] = states[name][n]
                for name, spec in self.actions_spec.items():
                    self.actions_buffers[name][parallel[n], index] = actions[name][n]
                    if spec['type'] == 'int':
                        name = name + '_mask'
                        if name in auxiliaries:
                            self.states_buffers[name][parallel[n], index] = auxiliaries[name][n]
                        else:
                            shape = (1,) + spec['shape'] + (spec['num_values'],)
                            self.states_buffers[name][parallel[n], index] = np.full(
                                shape=shape, fill_value=True, dtype=util.np_dtype(dtype='bool')
                            )

        # Reverse normalized actions dictionary
        actions = util.unpack_values(
            value_type='action', values=actions, values_spec=self.actions_spec
        )

        # Unbatch actions
        if batched:
            if isinstance(actions, dict):
                actions = [
                    OrderedDict(((name, actions[name][n]) for name in actions))
                    for n in range(len(parallel))
                ]
        else:
            actions = util.fmap(
                function=(lambda x: x[0]), xs=actions, depth=int(isinstance(actions, dict))
            )

        if query is None:
            return actions
        else:
            return actions, queried

    def observe(self, reward, terminal=False, parallel=0, query=None, **kwargs):
        """
        Observes reward and whether a terminal state is reached, needs to be preceded by
        `act(...)`.

        Args:
            reward (float | iter[float]): Reward
                (<span style="color:#C00000"><b>required</b></span>).
            terminal (bool | 0 | 1 | 2 | iter[...]): Whether a terminal state is reached or 2 if
                the episode was aborted (<span style="color:#00C000"><b>default</b></span>: false).
            parallel (int, iter[int]): Parallel execution index
                (<span style="color:#00C000"><b>default</b></span>: 0).
            query (list[str]): Names of tensors to retrieve
                (<span style="color:#00C000"><b>default</b></span>: none).
            kwargs: Additional input values, for instance, for dynamic hyperparameters.

        Returns:
            (bool | int, optional list[str]): Whether an update was performed, plus queried tensor
            values if requested.
        """
        assert util.reduce_all(predicate=util.not_nan_inf, xs=reward)

        if query is not None and self.parallel_interactions > 1:
            raise TensorforceError.invalid(
                name='agent.observe', argument='query', condition='parallel_interactions > 1'
            )

        batched = (not isinstance(parallel, int))
        if batched:
            if len(parallel) == 0:
                raise TensorforceError.value(
                    name='agent.observe', argument='parallel', value=parallel, hint='zero-length'
                )
        else:
            terminal = [terminal]
            reward = [reward]
            parallel = [parallel]

        num_updates = 0
        # TODO: Differently if not buffer_observe
        for terminal, reward, parallel in zip(terminal, reward, parallel):
            # Update terminal/reward buffer
            if isinstance(terminal, bool):
                terminal = int(terminal)
            index = self.buffer_indices[parallel]
            self.terminal_buffers[parallel, index] = terminal
            self.reward_buffers[parallel, index] = reward
            index += 1

            if self.max_episode_timesteps is not None and index > self.max_episode_timesteps:
                raise TensorforceError.value(
                    name='agent.observe', argument='index', value=index,
                    condition='> max_episode_timesteps'
                )

            if terminal > 0 or index == self.buffer_observe or query is not None:
                terminal = self.terminal_buffers[parallel, :index]
                reward = self.reward_buffers[parallel, :index]

                if self.recorder_spec is not None and \
                        self.episodes >= self.recorder_spec.get('start', 0):
                    for name in self.states_spec:
                        self.record_states[name].append(
                            np.array(self.states_buffers[name][parallel, :index])
                        )
                    for name, spec in self.actions_spec.items():
                        self.record_actions[name].append(
                            np.array(self.actions_buffers[name][parallel, :index])
                        )
                        if spec['type'] == 'int':
                            self.record_states[name + '_mask'].append(
                                np.array(self.states_buffers[name + '_mask'][parallel, :index])
                            )
                    self.record_terminal.append(np.array(terminal))
                    self.record_reward.append(np.array(reward))

                    if terminal[-1] > 0:
                        self.num_episodes += 1

                        if self.num_episodes == self.recorder_spec.get('frequency', 1):
                            directory = self.recorder_spec['directory']
                            if os.path.isdir(directory):
                                files = sorted(
                                    f for f in os.listdir(directory)
                                    if os.path.isfile(os.path.join(directory, f))
                                    and f.startswith('trace-')
                                )
                            else:
                                os.makedirs(directory)
                                files = list()
                            max_traces = self.recorder_spec.get('max-traces')
                            if max_traces is not None and len(files) > max_traces - 1:
                                for filename in files[:-max_traces + 1]:
                                    filename = os.path.join(directory, filename)
                                    os.remove(filename)

                            filename = 'trace-{}-{}.npz'.format(
                                self.episodes, time.strftime('%Y%m%d-%H%M%S')
                            )
                            filename = os.path.join(directory, filename)
                            self.record_states = util.fmap(
                                function=np.concatenate, xs=self.record_states, depth=1
                            )
                            self.record_actions = util.fmap(
                                function=np.concatenate, xs=self.record_actions, depth=1
                            )
                            self.record_terminal = np.concatenate(self.record_terminal)
                            self.record_reward = np.concatenate(self.record_reward)
                            np.savez_compressed(
                                filename, **self.record_states, **self.record_actions,
                                terminal=self.record_terminal, reward=self.record_reward
                            )
                            self.record_states = util.fmap(
                                function=(lambda x: list()), xs=self.record_states, depth=1
                            )
                            self.record_actions = util.fmap(
                                function=(lambda x: list()), xs=self.record_actions, depth=1
                            )
                            self.record_terminal = list()
                            self.record_reward = list()
                            self.num_episodes = 0

                # Model.observe()
                if query is None:
                    updated, self.episodes, self.updates = self.model.observe(
                        terminal=terminal, reward=reward, parallel=[parallel], **kwargs
                    )

                else:
                    updated, self.episodes, self.updates, queried = self.model.observe(
                        terminal=terminal, reward=reward, parallel=[parallel], query=query,
                        **kwargs
                    )

                # Reset buffer index
                self.buffer_indices[parallel] = 0

            else:
                # Increment buffer index
                self.buffer_indices[parallel] = index
                updated = False

            num_updates += int(updated)

        if batched:
            updated = num_updates
        else:
            assert num_updates <= 1
            updated = (num_updates == 1)

        if query is None:
            return updated
        else:
            return updated, queried

    def save(self, directory=None, filename=None, append_timestep=True):
        """
        Saves the current state of the agent.

        Args:
            directory (str): Agent directory
                (<span style="color:#00C000"><b>default</b></span>: directory specified for
                TensorFlow saver).
            filename (str): Agent filename
                (<span style="color:#00C000"><b>default</b></span>: filename specified for
                TensorFlow saver, or "agent").
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

        result = self.model.save(
            directory=directory, filename=filename, append_timestep=append_timestep
        )

        if directory is None:
            directory = self.model.saver_directory
        if filename is None:
            filename = 'agent'
        file = os.path.join(directory, filename + '.json')
        with open(file, 'w') as fp:
            json.dump(obj=self.spec, fp=fp, cls=NumpyJSONEncoder)

        return result

    def restore(self, directory=None, filename=None):
        """
        Restores the agent.

        Args:
            directory (str): Agent directory
                (<span style="color:#00C000"><b>default</b></span>: directory specified for
                TensorFlow saver).
            filename (str): Agent filename
                (<span style="color:#00C000"><b>default</b></span>: latest checkpoint in
                directory).
        """
        if not hasattr(self, 'model'):
            raise TensorforceError(message="Missing agent attribute model.")

        if not self.is_initialized:
            self.initialize()

        self.timesteps, self.episodes, self.updates = self.model.restore(
            directory=directory, filename=filename
        )

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
            raise TensorforceError.value(
                name='agent.get_output_tensors', argument='function', value=function
            )

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
            raise TensorforceError.value(
                name='agent.get_query_tensors', argument='function', value=function
            )

    def get_available_summaries(self):
        """
        Returns the summary labels provided by the agent.

        Returns:
            list[str]: Available summary labels.
        """
        return self.model.get_available_summaries()

    def should_stop(self):
        return self.model.monitored_session.should_stop()
