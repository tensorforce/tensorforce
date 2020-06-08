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
import tensorflow as tf

import tensorforce.agents
from tensorforce import util, TensorforceError


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
                if key == 'parallel_interactions':
                    assert agent.spec[key] >= value
                else:
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
            agent.update(kwargs)
            kwargs = dict(agent)
            agent = kwargs.pop('agent', kwargs.pop('type', 'default'))

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

            elif agent in tensorforce.agents.agents:
                # Keyword specification
                agent = tensorforce.agents.agents[agent]
                return Agent.create(agent=agent, environment=environment, **kwargs)

            else:
                raise TensorforceError.value(name='Agent.create', argument='agent', dtype=agent)

        else:
            raise TensorforceError.type(name='Agent.create', argument='agent', dtype=type(agent))

    @staticmethod
    def load(directory=None, filename=None, format=None, environment=None, **kwargs):
        """
        Restores an agent from a specification directory/file.

        Args:
            directory (str): Checkpoint directory
                (<span style="color:#00C000"><b>default</b></span>: current directory ".").
            filename (str): Checkpoint filename, with or without append and extension
                (<span style="color:#00C000"><b>default</b></span>: "agent").
            format ("tensorflow" | "numpy" | "hdf5" | "pb-actonly"): File format, "pb-actonly" loads
                an act-only agent based on a Protobuf model
                (<span style="color:#00C000"><b>default</b></span>: format matching directory and
                filename, required to be unambiguous).
            environment (Environment object): Environment which the agent is supposed to be trained
                on, environment-related arguments like state/action space specifications and
                maximum episode length will be extract if given
                (<span style="color:#00C000"><b>recommended</b></span> unless "pb-actonly" format).
            kwargs: Additional arguments, invalid for "pb-actonly" format.
        """
        if directory is None:
            # default directory: current directory "."
            directory = '.'

        if filename is None:
            # default filename: "agent"
            filename = 'agent'

        agent = os.path.join(directory, os.path.splitext(filename)[0] + '.json')
        if not os.path.isfile(agent) and agent[agent.rfind('-') + 1: -5].isdigit():
            agent = agent[:agent.rindex('-')] + '.json'
        if os.path.isfile(agent):
            with open(agent, 'r') as fp:
                agent = json.load(fp=fp)
            if 'agent' in kwargs:
                if 'agent' in agent and agent['agent'] != kwargs['agent']:
                    raise TensorforceError.value(
                        name='Agent.load', argument='agent', value=kwargs['agent']
                    )
                agent['agent'] = kwargs.pop('agent')
        else:
            agent = kwargs
            kwargs = dict()

        # Overwrite values
        if environment is not None and environment.max_episode_timesteps() is not None:
            if 'max_episode_timesteps' in kwargs:
                assert kwargs['max_episode_timesteps'] >= environment.max_episode_timesteps()
                agent['max_episode_timesteps'] = kwargs['max_episode_timesteps']
            else:
                agent['max_episode_timesteps'] = environment.max_episode_timesteps()
        if 'parallel_interactions' in kwargs and kwargs['parallel_interactions'] > 1:
            agent['parallel_interactions'] = kwargs['parallel_interactions']

        if format == 'pb-actonly':
            assert environment is None
            assert len(kwargs) == 0
            agent = ActonlyAgent(
                path=os.path.join(directory, os.path.splitext(filename)[0] + '.pb'),
                states=agent['states'], actions=agent['actions'], internals=agent.get('internals'),
                initial_internals=agent.get('initial_internals')
            )

        else:
            agent.pop('internals', None)
            agent.pop('initial_internals', None)
            agent = Agent.create(agent=agent, environment=environment, **kwargs)
            agent.restore(directory=directory, filename=filename, format=format)

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
                raise TensorforceError.collision(
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
        self.timestep_completed = np.ndarray(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='bool')
        )

        self.timesteps = 0
        self.episodes = 0
        self.updates = 0

        # Setup Model
        if not hasattr(self, 'model'):
            raise TensorforceError(message="Missing agent attribute model.")

        self.model.initialize()

        self.internals_spec = self.model.internals_spec
        self.auxiliaries_spec = self.model.auxiliaries_spec

        if self.model.saver_directory is not None:
            path = os.path.join(self.model.saver_directory, self.model.saver_filename + '.json')
            try:
                with open(path, 'w') as fp:
                    spec = OrderedDict(self.spec)
                    spec['internals'] = self.internals_spec
                    spec['initial_internals'] = self.initial_internals()
                    json.dump(obj=spec, fp=fp, cls=TensorforceJSONEncoder)
            except BaseException:
                try:
                    with open(path, 'w') as fp:
                        spec = OrderedDict()
                        spec['states'] = self.spec['states']
                        spec['actions'] = self.spec['actions']
                        spec['internals'] = self.internals_spec
                        spec['initial_internals'] = self.initial_internals()
                        json.dump(obj=spec, fp=fp, cls=TensorforceJSONEncoder)
                except BaseException:
                    os.remove(path)
                    raise

        self.reset()

    def close(self):
        """
        Closes the agent.
        """
        self.model.close()
        self.model = None

    def reset(self):
        """
        Resets all agent buffers and discards unfinished episodes.
        """
        self.buffer_indices = np.zeros(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='int')
        )
        self.timestep_completed = np.ones(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='bool')
        )

        self.timesteps, self.episodes, self.updates = self.model.reset()

    def initial_internals(self):
        """
        Returns the initial internal agent state(s), to be used at the beginning of an episode as
        `internals` argument for `act(...)` in independent mode

        Returns:
            dict[internal]: Dictionary containing initial internal agent state(s).
        """
        return OrderedDict(**self.model.internals_init)

    def act(
        self, states, internals=None, parallel=0, independent=False, deterministic=False,
        evaluation=False, query=None, **kwargs
    ):
        """
        Returns action(s) for the given state(s), needs to be followed by `observe(...)` unless
        independent mode set via `independent`/`evaluation`.

        Args:
            states (dict[state] | iter[dict[state]]): Dictionary containing state(s) to be acted on
                (<span style="color:#C00000"><b>required</b></span>).
            internals (dict[internal] | iter[dict[internal]]): Dictionary containing current
                internal agent state(s), either given by `initial_internals()` at the beginning of
                an episode or as return value of the preceding `act(...)` call
                (<span style="color:#C00000"><b>required</b></span> if independent mode and agent
                has internal states).
            parallel (int | iter[int]): Parallel execution index
                (<span style="color:#00C000"><b>default</b></span>: 0).
            independent (bool): Whether act is not part of the main agent-environment interaction,
                and this call is thus not followed by observe
                (<span style="color:#00C000"><b>default</b></span>: false).
            deterministic (bool): Ff independent mode, whether to act deterministically, so no
                exploration and sampling
                (<span style="color:#00C000"><b>default</b></span>: false).
            evaluation (bool): Whether the agent is currently evaluated, implies independent and
                deterministic
                (<span style="color:#00C000"><b>default</b></span>: false).
            query (list[str]): Names of tensors to retrieve
                (<span style="color:#00C000"><b>default</b></span>: none).
            kwargs: Additional input values, for instance, for dynamic hyperparameters.

        Returns:
            dict[action] | iter[dict[action]], dict[internal] | iter[dict[internal]] if `internals`
            argument given, plus optional list[str]: Dictionary containing action(s), dictionary
            containing next internal agent state(s) if independent mode, plus queried tensor values
            if requested.
        """
        assert util.reduce_all(predicate=util.not_nan_inf, xs=states)

        if evaluation:
            if deterministic:
                raise TensorforceError.invalid(
                    name='agent.act', argument='deterministic', condition='evaluation = true'
                )
            if independent:
                raise TensorforceError.invalid(
                    name='agent.act', argument='independent', condition='evaluation = true'
                )
            deterministic = independent = True

        if not independent:
            if internals is not None:
                raise TensorforceError.invalid(
                    name='agent.act', argument='internals', condition='independent = false'
                )
            if deterministic:
                raise TensorforceError.invalid(
                    name='agent.act', argument='deterministic', condition='independent = false'
                )

        if independent:
            internals_is_none = (internals is None)
            if internals_is_none:
                if len(self.model.internals_spec) > 0:
                    raise TensorforceError.required(
                        name='agent.act', argument='internals', condition='independent = true'
                    )
                internals = OrderedDict()

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
            if independent:
                internals = OrderedDict((
                    (name, np.asarray([internals[n][name] for n in range(len(parallel))]))
                    for name in internals[0]
                ))
        else:
            parallel = np.asarray([parallel])
            states = util.fmap(
                function=(lambda x: np.asarray([x])), xs=states,
                depth=int(isinstance(states, dict))
            )
            if independent:
                internals = util.fmap(function=(lambda x: np.asarray([x])), xs=internals, depth=1)

        if not independent and not all(self.timestep_completed[n] for n in parallel):
            raise TensorforceError(message="Calling agent.act must be preceded by agent.observe.")

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
        if independent:
            if query is None:
                actions, internals = self.model.independent_act(
                    states=states, internals=internals, auxiliaries=auxiliaries, parallel=parallel,
                    deterministic=deterministic, **kwargs
                )

            else:
                actions, internals, queried = self.model.independent_act(
                    states=states, internals=internals, auxiliaries=auxiliaries, parallel=parallel,
                    deterministic=deterministic, query=query, **kwargs
                )

        else:
            if query is None:
                actions, self.timesteps = self.model.act(
                    states=states, auxiliaries=auxiliaries, parallel=parallel, **kwargs
                )

            else:
                actions, self.timesteps, queried = self.model.act(
                    states=states, auxiliaries=auxiliaries, parallel=parallel, query=query, **kwargs
                )

        if not independent:
            for n in parallel:
                self.timestep_completed[n] = False

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
            if independent:
                internals = util.fmap(function=(lambda x: x[0]), xs=internals, depth=1)

        if independent and not internals_is_none:
            if query is None:
                return actions, internals
            else:
                return actions, internals, queried

        else:
            if independent and len(internals) > 0:
                raise TensorforceError.unexpected()
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
            if query is not None:
                raise TensorforceError.invalid(
                    name='agent.observe', argument='query', condition='len(parallel) > 1'
                )
        else:
            terminal = [terminal]
            reward = [reward]
            parallel = [parallel]

        if any(self.timestep_completed[n] for n in parallel):
            raise TensorforceError(message="Calling agent.observe must be preceded by agent.act.")

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
            self.buffer_indices[parallel] = index

            if self.max_episode_timesteps is not None and index > self.max_episode_timesteps:
                raise TensorforceError.value(
                    name='agent.observe', argument='index', value=index,
                    condition='> max_episode_timesteps'
                )

            if terminal > 0 or index == self.buffer_observe or query is not None:
                self.timestep_completed[parallel] = True
                if query is None:
                    updated = self.model_observe(parallel=parallel, **kwargs)
                else:
                    updated, queried = self.model_observe(parallel=parallel, query=query, **kwargs)

            else:
                # Increment buffer index
                self.timestep_completed[parallel] = True
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

    def model_observe(self, parallel, query=None, **kwargs):
        assert self.timestep_completed[parallel]
        index = self.buffer_indices[parallel]
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

        # Reset buffer index
        self.buffer_indices[parallel] = 0

        # Model.observe()
        if query is None:
            updated, self.episodes, self.updates = self.model.observe(
                terminal=terminal, reward=reward, parallel=[parallel], **kwargs
            )
            return updated

        else:
            updated, self.episodes, self.updates, queried = self.model.observe(
                terminal=terminal, reward=reward, parallel=[parallel], query=query,
                **kwargs
            )
            return updated, queried

    def save(self, directory=None, filename=None, format='tensorflow', append=None):
        """
        Saves the agent to a checkpoint.

        Args:
            directory (str): Checkpoint directory
                (<span style="color:#00C000"><b>default</b></span>: directory specified for
                TensorFlow saver, otherwise current directory).
            filename (str): Checkpoint filename, without extension
                (<span style="color:#00C000"><b>default</b></span>: filename specified for
                TensorFlow saver, otherwise name of agent).
            format ("tensorflow" | "numpy" | "hdf5"): File format, "tensorflow" uses TensorFlow
                saver to store variables, graph meta information and an optimized Protobuf model
                with an act-only graph, whereas the others only store variables as NumPy/HDF5 file
                (<span style="color:#00C000"><b>default</b></span>: TensorFlow format).
            append ("timesteps" | "episodes" | "updates"): Append current timestep/episode/update to
                checkpoint filename
                (<span style="color:#00C000"><b>default</b></span>: none).

        Returns:
            str: Checkpoint path.
        """
        # TODO: Messes with required parallel disentangling, better to remove unfinished episodes
        # from memory, but currently entire episode buffered anyway...
        # Empty buffers before saving
        for parallel in range(self.parallel_interactions):
            if self.buffer_indices[parallel] > 0:
                self.model_observe(parallel=parallel)

        if directory is None:
            # default directory: saver if given, otherwise current directory "."
            if self.model.saver_directory is None:
                directory = '.'
            else:
                directory = self.model.saver_directory

        if filename is None:
            # default filename: saver which defaults to agent name
            filename = self.model.saver_filename

        path = self.model.save(directory=directory, filename=filename, format=format, append=append)

        spec_path = os.path.join(directory, filename + '.json')
        try:
            with open(spec_path, 'w') as fp:
                spec = OrderedDict(self.spec)
                spec['internals'] = self.internals_spec
                spec['initial_internals'] = self.initial_internals()
                json.dump(obj=spec, fp=fp, cls=TensorforceJSONEncoder)
        except BaseException:
            try:
                with open(spec_path, 'w') as fp:
                    spec = OrderedDict()
                    spec['states'] = self.spec['states']
                    spec['actions'] = self.spec['actions']
                    spec['internals'] = self.internals_spec
                    spec['initial_internals'] = self.initial_internals()
                    json.dump(obj=spec, fp=fp, cls=TensorforceJSONEncoder)
            except BaseException:
                os.remove(spec_path)

        return path

    def restore(self, directory=None, filename=None, format=None):
        """
        Restores the agent from a checkpoint.

        Args:
            directory (str): Checkpoint directory
                (<span style="color:#00C000"><b>default</b></span>: directory specified for
                TensorFlow saver, otherwise current directory).
            filename (str): Checkpoint filename, with or without append and extension
                (<span style="color:#00C000"><b>default</b></span>: filename specified for
                TensorFlow saver, otherwise name of agent or latest checkpoint in directory).
            format ("tensorflow" | "numpy" | "hdf5"): File format
                (<span style="color:#00C000"><b>default</b></span>: format matching directory and
                filename, required to be unambiguous).
        """
        if not hasattr(self, 'model'):
            raise TensorforceError(message="Missing agent attribute model.")

        if not self.is_initialized:
            self.initialize()

        if directory is None:
            # default directory: saver if given, otherwise current directory "."
            if self.model.saver_directory is None:
                directory = '.'
            else:
                directory = self.model.saver_directory

        if filename is None:
            # default filename: saver which defaults to agent name
            filename = self.model.saver_filename

        # format implicitly given if file exists
        if format is None and os.path.isfile(os.path.join(directory, filename)):
            if '.data-' in filename:
                filename = filename[:filename.index('.data-')]
                format = 'tensorflow'
            elif filename.endswith('.npz'):
                filename = filename[:-4]
                format = 'numpy'
            elif filename.endswith('.hdf5'):
                filename = filename[:-5]
                format = 'hdf5'
            elif filename.endswith('.h5'):
                filename = filename[:-3]
                format = 'hdf5'
            else:
                assert False
        elif format is None and os.path.isfile(os.path.join(directory, filename + '.meta')):
            format = 'tensorflow'
        elif format is None and os.path.isfile(os.path.join(directory, filename + '.npz')):
            format = 'numpy'
        elif format is None and (
            os.path.isfile(os.path.join(directory, filename + '.hdf5')) or
            os.path.isfile(os.path.join(directory, filename + '.h5'))
        ):
            format = 'hdf5'

        else:
            # infer format from directory
            found = None
            latest = -1
            for name in os.listdir(directory):
                if format in (None, 'numpy') and name == filename + '.npz':
                    assert found is None
                    found = 'numpy'
                    latest = None
                elif format in (None, 'numpy') and name.startswith(filename) and \
                        name.endswith('.npz'):
                    assert found is None or found == 'numpy'
                    found = 'numpy'
                    n = int(name[len(filename) + 1: -4])
                    if n > latest:
                        latest = n
                elif format in (None, 'hdf5') and \
                        (name == filename + '.hdf5' or  name == filename + '.h5'):
                    assert found is None
                    found = 'hdf5'
                    latest = None
                elif format in (None, 'hdf5') and name.startswith(filename) and \
                        (name.endswith('.hdf5') or name.endswith('.h5')):
                    assert found is None or found == 'hdf5'
                    found = 'hdf5'
                    n = int(name[len(filename) + 1: -5])
                    if n > latest:
                        latest = n

            if latest == -1:
                if format is None:
                    format = 'tensorflow'
                else:
                    assert format == 'tensorflow'
                if filename is None or not os.path.isfile(os.path.join(directory, filename + '.meta')):
                    path = tf.compat.v1.train.latest_checkpoint(checkpoint_dir=directory, latest_filename=None)
                    if '/' in path:
                        filename = path[path.rindex('/') + 1:]
                    else:
                        filename = path

            else:
                if format is None:
                    format = found
                else:
                    assert format == found
                if latest is not None:
                    filename = filename + '-' + str(latest)

        self.timesteps, self.episodes, self.updates = self.model.restore(
            directory=directory, filename=filename, format=format
        )

    def get_variables(self):
        """
        Returns the names of all agent variables.

        Returns:
            list[str]: Names of variables.
        """
        return [
            variable.name[len(self.model.name) + 1: -2] for variable in self.model.get_variables()
        ]

    def get_variable(self, variable):
        """
        Returns the value of the variable with the given name.

        Args:
            variable (string): Variable name
                (<span style="color:#C00000"><b>required</b></span>).

        Returns:
            numpy-array: Variable value.
        """
        return self.model.get_variable(variable=variable)

    def assign_variable(self, variable, value):
        """
        Assigns the given value to the variable with the given name.

        Args:
            variable (string): Variable name
                (<span style="color:#C00000"><b>required</b></span>).
            value (variable-compatible value): Value to assign to variable
                (<span style="color:#C00000"><b>required</b></span>).
        """
        self.model.assign_variable(variable=variable, value=value)

    def summarize(self, summary, value, step=None):
        """
        Records a value for the given custom summary label (as specified via summarizer[custom]).

        Args:
            variable (string): Custom summary label
                (<span style="color:#C00000"><b>required</b></span>).
            value (summary-compatible value): Summary value to record
                (<span style="color:#C00000"><b>required</b></span>).
            step (int): Summary recording step
                (<span style="color:#00C000"><b>default</b></span>: current timestep).
        """
        self.model.summarize(summary=summary, value=value, step=step)

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

    def get_available_summaries(self):
        """
        Returns the summary labels provided by the agent.

        Returns:
            list[str]: Available summary labels.
        """
        return self.model.get_available_summaries()

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


class ActonlyAgent(object):

    def __init__(self, path, states, actions, internals=None, initial_internals=None):
        self.states_spec = states
        self.actions_spec = actions
        if internals is None:
            assert initial_internals is None
            self.internals_spec = OrderedDict()
            self._initial_internals = OrderedDict()
        else:
            assert list(internals) == list(initial_internals)
            self.internals_spec = internals
            self._initial_internals = initial_internals

        with tf.io.gfile.GFile(name=path, mode='rb') as filehandle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(filehandle.read())
        graph = tf.Graph()
        with graph.as_default():
            tf.graph_util.import_graph_def(graph_def=graph_def, name='')
        graph.finalize()
        self.session = tf.compat.v1.Session(graph=graph)
        self.session.__enter__()

    def close(self):
        self.session.__exit__(None, None, None)
        tf.compat.v1.reset_default_graph()

    def initial_internals(self):
        return OrderedDict(**self._initial_internals)

    def act(
        self, states, internals=None, parallel=0, independent=True, deterministic=True,
        evaluation=True, query=None, **kwargs
    ):
        # Invalid arguments
        assert parallel == 0 and independent and deterministic and evaluation and \
            query is None and len(kwargs) == 0

        assert util.reduce_all(predicate=util.not_nan_inf, xs=states)
        internals_is_none = (internals is None)
        if internals_is_none:
            if len(self.internals_spec) > 0:
                raise TensorforceError.required(name='agent.act', argument='internals')
            internals = OrderedDict()

        # Batch states
        name = next(iter(self.states_spec))
        batched = (np.asarray(states[name]).ndim > len(self.states_spec[name]['shape']))
        if batched:
            if isinstance(states[0], dict):
                states = OrderedDict((
                    (name, np.asarray([states[n][name] for n in range(len(parallel))]))
                    for name in states[0]
                ))
            else:
                states = np.asarray(states)
            internals = OrderedDict((
                (name, np.asarray([internals[n][name] for n in range(len(parallel))]))
                for name in internals[0]
            ))
        else:
            states = util.fmap(
                function=(lambda x: np.asarray([x])), xs=states,
                depth=int(isinstance(states, dict))
            )
            internals = util.fmap(function=(lambda x: np.asarray([x])), xs=internals, depth=1)

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
        fetches = (
            {
                name: util.join_scopes('agent.independent_act', name + '-output:0')
                for name in self.actions_spec
            }, {
                name: util.join_scopes('agent.independent_act', name + '-output:0')
                for name in self.internals_spec
            }
        )

        feed_dict = dict()
        for name, state in states.items():
            feed_dict[util.join_scopes('agent', name + '-input:0')] = state
        for name, auxiliary in auxiliaries.items():
            feed_dict[util.join_scopes('agent', name + '-input:0')] = auxiliary
        for name, internal in internals.items():
            feed_dict[util.join_scopes('agent', name + '-input:0')] = internal

        actions, internals = self.session.run(fetches=fetches, feed_dict=feed_dict)

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
            internals = util.fmap(function=(lambda x: x[0]), xs=internals, depth=1)

        if internals_is_none:
            if len(internals) > 0:
                raise TensorforceError.unexpected()
            return actions
        else:
            return actions, internals


class TensorforceJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder which is NumPy-compatible.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)
