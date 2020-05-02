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
from tensorforce.core import ArrayDict, NestedDict, TensorforceConfig


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
        self, states, actions, max_episode_timesteps=None, parallel_interactions=1, config=None,
        recorder=None
    ):
        self.is_initialized = False

        # Check whether spec attribute exists
        if not hasattr(self, 'spec'):
            raise TensorforceError.required_attribute(name='Agent', attribute='spec')

        # States/actions, plus single state/action flag
        if 'shape' in states:
            self.states_spec = dict(state=states)
            self.single_state = True
        else:
            self.states_spec = states
            self.single_state = False
        if 'type' in actions:
            self.actions_spec = dict(action=actions)
            self.single_action = True
        else:
            self.actions_spec = actions
            self.single_action = False

        # Max episode timesteps
        self.max_episode_timesteps = max_episode_timesteps

        # Parallel interactions
        if isinstance(parallel_interactions, int):
            if parallel_interactions <= 0:
                raise TensorforceError.value(
                    name='Agent', argument='parallel_interactions', value=parallel_interactions,
                    hint='<= 0'
                )
            self.parallel_interactions = parallel_interactions
        else:
            raise TensorforceError.type(
                name='Agent', argument='parallel_interactions', dtype=type(parallel_interactions)
            )

        # Buffer observe
        if config is None:
            config = dict()
        buffer_observe = config.get('buffer_observe', True)
        if isinstance(buffer_observe, bool):
            if self.parallel_interactions > 1:
                if not buffer_observe:
                    raise TensorforceError.required(
                        name='Agent', argument='config.buffer_observe',
                        condition='parallel_interactions > 1'
                    )
                elif self.max_episode_timesteps is None:
                    raise TensorforceError.required(
                        name='Agent', argument='max_episode_timesteps',
                        condition='parallel_interactions > 1'
                    )
            if not buffer_observe:
                buffer_observe = 1
            elif self.max_episode_timesteps is None:
                buffer_observe = 100
            else:
                buffer_observe = self.max_episode_timesteps
        elif isinstance(buffer_observe, int):
            if buffer_observe <= 0:
                raise TensorforceError.value(
                    name='Agent', argument='config.buffer_observe', value=buffer_observe,
                    hint='<= 0'
                )
            if self.parallel_interactions > 1:
                raise TensorforceError.value(
                    name='Agent', argument='config.buffer_observe', value=buffer_observe,
                    condition='parallel_interactions > 1'
                )
            if self.max_episode_timesteps is None:
                buffer_observe = buffer_observe
            else:
                buffer_observe = min(buffer_observe, self.max_episode_timesteps)
        else:
            raise TensorforceError.type(
                name='Agent', argument='config.buffer_observe', dtype=type(buffer_observe)
            )

        # Tensorforce config
        config['buffer_observe'] = buffer_observe
        self.config = TensorforceConfig(**config)

        # Random seed
        if self.config.seed is not None:
            random.seed(a=self.config.seed)
            np.random.seed(seed=self.config.seed)
            tf.random.set_seed(seed=self.config.seed)

        # Recorder
        if recorder is None:
            pass
        elif not all(key in ('directory', 'frequency', 'max-traces', 'start') for key in recorder):
            raise TensorforceError.value(
                name='Agent', argument='recorder values', value=list(recorder),
                hint='not from {directory,frequency,max-traces,start}'
            )
        self.recorder_spec = recorder if recorder is None else dict(recorder)

    def __str__(self):
        return self.__class__.__name__

    def initialize(self):
        """
        Initializes the agent, usually done as part of Agent.create() / Agent.load().
        """
        # Check whether already initialized
        if self.is_initialized:
            raise TensorforceError(
                message="Agent is already initialized, possibly as part of Agent.create()."
            )
        self.is_initialized = True

        # Initialize model
        if not hasattr(self, 'model'):
            raise TensorforceError.required_attribute(name='Agent', attribute='model')
        self.model.root_initialize()

        # Value space specifications
        self.states_spec = self.model.unprocessed_states_spec
        self.internals_spec = self.model.internals_spec
        self.auxiliaries_spec = self.model.auxiliaries_spec
        self.actions_spec = self.model.actions_spec
        self.terminal_spec = self.model.terminal_spec
        self.reward_spec = self.model.reward_spec
        self.parallel_spec = self.model.parallel_spec

        # Parallel buffer indices
        self.buffer_indices = np.zeros(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='int')
        )
        self.timestep_completed = np.ones(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='bool')
        )

        # Parallel terminal/reward buffers
        self.buffers = ArrayDict()
        shape = (self.parallel_interactions, self.config.buffer_observe)
        self.buffers['terminal'] = np.ndarray(
            shape=(shape + self.terminal_spec.shape), dtype=self.terminal_spec.np_type()
        )
        self.buffers['reward'] = np.ndarray(
            shape=(shape + self.reward_spec.shape), dtype=self.reward_spec.np_type()
        )

        # Recorder buffers if required
        if self.recorder_spec is not None:
            self.num_episodes = 0
            self.recorded = NestedDict(value_type=list, overwrite=False)
            for name, spec in self.states_spec.items():
                self.buffers[name] = np.ndarray(shape=(shape + spec.shape), dtype=spec.np_type())
                self.recorded[name] = list()
            for name, spec in self.auxiliaries_spec.items():
                self.buffers[name] = np.ndarray(shape=(shape + spec.shape), dtype=spec.np_type())
                self.recorded[name] = list()
            for name, spec in self.actions_spec.items():
                self.buffers[name] = np.ndarray(shape=(shape + spec.shape), dtype=spec.np_type())
                self.recorded[name] = list()
            self.recorded['terminal'] = list()
            self.recorded['reward'] = list()

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

        # Reset model
        self.timesteps, self.episodes, self.updates = self.model.reset()

    def close(self):
        """
        Closes the agent.
        """
        self.model.close()
        del self.model

    def reset(self):
        """
        Resets all agent buffers and discards unfinished episodes.
        """
        # Reset parallel buffer indices
        self.buffer_indices = np.zeros(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='int')
        )
        self.timestep_completed = np.ones(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='bool')
        )

        # Reset model
        self.timesteps, self.episodes, self.updates = self.model.reset()

    def initial_internals(self):
        """
        Returns the initial internal agent state(s), to be used at the beginning of an episode as
        `internals` argument for `act(...)` in independent mode

        Returns:
            dict[internal]: Dictionary containing initial internal agent state(s).
        """
        return self.model.internals_init.copy()

    def act(self, states, internals=None, parallel=0, independent=False):
        """
        Returns action(s) for the given state(s), needs to be followed by `observe()` unless
        independent mode.

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

        Returns:
            dict[action] | iter[dict[action]], dict[internal] | iter[dict[internal]] if `internals`
            argument given: Dictionary containing action(s), dictionary containing next internal
            agent state(s) if independent mode.
        """
        # Independent and internals
        if independent:
            if parallel != 0:
                raise TensorforceError.invalid(
                    name='Agent.act', argument='parallel', condition='independent is true'
                )
            is_internals_none = (internals is None)
            if is_internals_none and len(self.model.internals_spec) > 0:
                raise TensorforceError.required(
                    name='Agent.act', argument='internals', condition='independent is true'
                )
        else:
            if internals is not None:
                raise TensorforceError.invalid(
                    name='Agent.act', argument='internals', condition='independent is false'
                )

        # Check whether input is batched based on parallel argument
        if self.single_state and not isinstance(states, dict) and not (
            util.is_iterable(x=states) and isinstance(states[0], dict)
        ):
            states = np.asarray(states)
            if states.shape == self.states_spec['state'].shape:
                # Single state is not batched
                states = ArrayDict(state=np.expand_dims(states, axis=0))
                batched = False
                num_parallel = 1
                states

            else:
                # Single state is batched, iter[state]
                assert states.shape[1:] == self.states_spec['state'].shape
                states = ArrayDict(state=states)
                batched = True
                num_parallel = states.shape[0]

            if independent:
                # Independent mode: handle internals argument
                if is_internals_none:
                    internals = ArrayDict()
                elif util.is_iterable(x=internals):
                    is_iter_of_dicts = True
                    if not batched:
                        raise TensorforceError.type(
                            name='Agent.act', argument='internals', dtype=type(internals),
                            hint='is batched'
                        )
                    elif len(internals) != num_parallel:
                        raise TensorforceError.value(
                            name='Agent.act', argument='len(internals)', value=len(internals),
                            hint='!= len(states)'
                        )
                    else:
                        for n, x in enumerate(internals):
                            if not isinstance(x, dict):
                                raise TensorforceError.type(
                                    name='Agent.act', argument='internals[{}]'.format(n),
                                    dtype=type(x), hint='is not dict'
                                )
                        # Turn iter of dicts into dict of arrays (TODO: recursive)
                        internals = ArrayDict((
                            (name, np.asarray([x[name] for x in internals]))
                            for name in internals[0]
                        ))

                elif isinstance(internals, dict):
                    # Turn into arrays (TODO: recursive)
                    if batched:
                        internals = ArrayDict((
                            (name, np.asarray(internal)) for name, internal in internals.items()
                        ))
                    else:
                        internals = ArrayDict((
                            (name, np.asarray([internal]))
                            for name, internal in internals.items()
                        ))

                else:
                    raise TensorforceError.type(
                        name='Agent.act', argument='internals', dtype=type(internals),
                        hint='is not iterable/dict'
                    )

            else:
                # Non-independent mode: handle parallel input
                if parallel == 0:
                    if batched:
                        assert num_parallel == self.parallel_interactions
                        parallel = np.asarray(list(range(num_parallel)))
                    else:
                        parallel = np.asarray([parallel])
                elif not util.is_iterable(x=parallel):
                    raise TensorforceError.type(
                        name='Agent.act', argument='parallel', dtype=type(parallel),
                        hint='is not iterable'
                    )
                elif len(parallel) != num_parallel:
                    raise TensorforceError.value(
                        name='Agent.act', argument='len(parallel)', value=len(parallel),
                        hint='!= len(states)'
                    )
                else:
                    parallel = np.asarray(parallel)

        else:
            if util.is_iterable(x=states):
                # States is batched, iter[dict[state]]
                batched = True
                is_iter_of_dicts = True
                num_parallel = len(states)
                if num_parallel == 0:
                    raise TensorforceError.value(
                        name='Agent.act', argument='len(states)', value=num_parallel, hint='= 0'
                    )
                for n, x in enumerate(states):
                    if not isinstance(x, dict):
                        raise TensorforceError.type(
                            name='Agent.act', argument='states[{}]'.format(n), dtype=type(x),
                            hint='is not dict'
                        )
                # Turn iter of dicts into dict of arrays (TODO: recursive)
                # Doesn't use self.states_spec since states also contains auxiliaries
                states = ArrayDict((
                    (name, np.asarray([x[name] for x in states])) for name in states[0]
                ))

                if independent:
                    # Independent mode: handle internals argument
                    if is_internals_none:
                        internals = ArrayDict()
                    elif not util.is_iterable(x=internals):
                        raise TensorforceError.type(
                            name='Agent.act', argument='internals', dtype=type(internals),
                            hint='is not iterable'
                        )
                    elif len(internals) != num_parallel:
                        raise TensorforceError.value(
                            name='Agent.act', argument='len(internals)', value=len(internals),
                            hint='!= len(states)'
                        )
                    else:
                        for n, x in enumerate(internals):
                            if not isinstance(x, dict):
                                raise TensorforceError.type(
                                    name='Agent.act', argument='internals[{}]'.format(n),
                                    dtype=type(x), hint='is not dict'
                                )
                        # Turn iter of dicts into dict of arrays (TODO: recursive)
                        internals = ArrayDict((
                            (name, np.asarray([x[name] for x in internals]))
                            for name in internals[0]
                        ))

                else:
                    # Non-independent mode: handle parallel input
                    if parallel == 0:
                        if num_parallel == 1:
                            parallel = np.asarray([parallel])
                        else:
                            assert num_parallel == self.parallel_interactions
                            parallel = np.asarray(list(range(num_parallel)))
                    elif not util.is_iterable(x=parallel):
                        raise TensorforceError.type(
                            name='Agent.act', argument='parallel', dtype=type(parallel),
                            hint='is not iterable'
                        )
                    elif len(parallel) != num_parallel:
                        raise TensorforceError.value(
                            name='Agent.act', argument='len(parallel)', value=len(parallel),
                            hint='!= len(states)'
                        )
                    else:
                        parallel = np.asarray(parallel)

            elif isinstance(states, dict):
                # States is dict, turn into arrays (TODO: recursive)
                states = ArrayDict(((name, np.asarray(state)) for name, state in states.items()))
                name, spec = self.states_spec.item()

                if states[name].shape == spec.shape:
                    # States is not batched, dict[state]
                    states = states.fmap(function=(lambda state: np.expand_dims(state, axis=0)))
                    batched = False
                    num_parallel = 1

                else:
                    # States is batched, dict[iter[state]]
                    assert states[name].shape[1:] == spec.shape
                    batched = True
                    is_iter_of_dicts = False
                    num_parallel = states[name].shape[0]
                    if num_parallel == 0:
                        raise TensorforceError.value(
                            name='Agent.act', argument='len(states)', value=num_parallel, hint='= 0'
                        )

                if independent:
                    # Independent mode: handle internals argument
                    if is_internals_none:
                        internals = ArrayDict()
                    elif not isinstance(internals, dict):
                        raise TensorforceError.type(
                            name='Agent.act', argument='internals', dtype=type(internals),
                            hint='is not dict'
                        )
                    else:
                        # Turn into arrays (TODO: recursive)
                        if batched:
                            internals = ArrayDict((
                                (name, np.asarray(internal)) for name, internal in internals.items()
                            ))
                        else:
                            internals = ArrayDict((
                                (name, np.asarray([internal]))
                                for name, internal in internals.items()
                            ))

                else:
                    # Non-independent mode: handle parallel input
                    if parallel == 0:
                        if batched:
                            assert num_parallel == self.parallel_interactions
                            parallel = np.asarray(list(range(num_parallel)))
                        else:
                            parallel = np.asarray([parallel])
                    elif not util.is_iterable(x=parallel):
                        raise TensorforceError.type(
                            name='Agent.act', argument='parallel', dtype=type(parallel),
                            hint='is not iterable'
                        )
                    elif len(parallel) != num_parallel:
                        raise TensorforceError.value(
                            name='Agent.act', argument='len(parallel)', value=len(parallel),
                            hint='!= len(states)'
                        )
                    else:
                        parallel = np.asarray(parallel)

            else:
                raise TensorforceError.type(
                    name='Agent.act', argument='states', dtype=type(states),
                    hint='is not iterable/dict'
                )

        # If not independent, check whether previous timesteps were completed
        if not independent:
            if not self.timestep_completed[parallel].all():
                raise TensorforceError(
                    message="Calling agent.act must be preceded by agent.observe."
                )
            self.timestep_completed[parallel] = False

        def function(name, spec):
            auxiliary = ArrayDict()
            if self.config.enable_int_action_masking and spec.type == 'int' and \
                    spec.num_values is not None:
                # Mask, either part of states or default all true
                auxiliary['mask'] = np.asarray(states.pop(name + '_mask', np.ones(
                    shape=(num_parallel,) + spec.shape + (spec.num_values,), dtype=spec.np_type()
                )))
            return auxiliary

        auxiliaries = self.actions_spec.fmap(function=function, cls=ArrayDict, with_names=True)

        # Buffer inputs for recording
        if self.recorder_spec is not None and not independent and \
                self.episodes >= self.recorder_spec.get('start', 0):
            for n in range(num_parallel):
                index = self.buffer_indices[parallel[n]]
                for name in self.states_spec:
                    self.buffers[name][parallel[n], index] = states[name][n]
                for name in self.auxiliaries_spec:
                    self.buffers[name][parallel[n], index] = auxiliaries[name][n]

        # Inputs to tensors
        states = self.states_spec.to_tensor(value=states, batched=True)
        if independent:
            internals = self.internals_spec.to_tensor(value=internals, batched=True)
        auxiliaries = self.auxiliaries_spec.to_tensor(value=auxiliaries, batched=True)
        parallel_tensor = self.parallel_spec.to_tensor(value=parallel, batched=True)

        # Model.act()
        if independent:
            actions, internals = self.model.independent_act(
                states=states, internals=internals, auxiliaries=auxiliaries
            )
            assert not is_internals_none or len(internals) == 0
        else:
            actions, self.timesteps = self.model.act(
                states=states, auxiliaries=auxiliaries, parallel=parallel_tensor
            )

        # Outputs from tensors
        actions = self.actions_spec.from_tensor(tensor=actions, batched=True)

        # Buffer outputs for recording
        if self.recorder_spec is not None and not independent and \
                self.episodes >= self.recorder_spec.get('start', 0):
            for n in range(num_parallel):
                index = self.buffer_indices[parallel[n]]
                for name in self.actions_spec:
                    self.buffers[name][parallel[n], index] = actions[name][n]

        # Unbatch actions
        if batched:
            # If inputs were batched, turn list of dicts into dict of lists
            if self.single_action:
                actions = actions['action']
            else:
                # TODO: recursive
                actions = [
                    OrderedDict(((name, x[n]) for name, x in actions.items()))
                    for n in range(num_parallel)
                ]

            if independent and not is_internals_none and is_iter_of_dicts:
                # TODO: recursive
                internals = [
                    OrderedDict(((name, x[n]) for name, x in internals.items()))
                    for n in range(num_parallel)
                ]

        else:
            # If inputs were not batched, unbatch outputs
            function = (lambda x: x.item() if x.shape == (1,) else x[0])
            if self.single_action:
                actions = function(actions['action'])
            else:
                actions = util.fmap(function=function, xs=actions, map_types=(tuple, list))
            if independent:
                internals = util.fmap(function=function, xs=internals, map_types=(tuple, list))

        if independent and not is_internals_none:
            return actions, internals
        else:
            return actions

    def observe(self, reward=0.0, terminal=False, parallel=0):
        """
        Observes reward and whether a terminal state is reached, needs to be preceded by `act()`.

        Args:
            reward (float | iter[float]): Reward
                (<span style="color:#00C000"><b>default</b></span>: 0.0).
            terminal (bool | 0 | 1 | 2 | iter[...]): Whether a terminal state is reached, or 2 if
                the episode was aborted
                (<span style="color:#00C000"><b>default</b></span>: false).
            parallel (int, iter[int]): Parallel execution index
                (<span style="color:#00C000"><b>default</b></span>: 0).

        Returns:
            int: Number of performed updates.
        """
        # Check whether inputs are batched
        if util.is_iterable(x=reward):
            reward = np.asarray(reward)
            num_parallel = reward.shape[0]
            if terminal is False:
                terminal = np.asarray([0 for _ in range(num_parallel)])
            if parallel == 0:
                assert num_parallel == self.parallel_interactions
                parallel = np.asarray(list(range(num_parallel)))

        elif util.is_iterable(x=terminal):
            terminal = np.asarray([int(t) for t in terminal])
            num_parallel = terminal.shape[0]
            if reward == 0.0:
                reward = np.asarray([0.0 for _ in range(num_parallel)])
            if parallel == 0:
                assert num_parallel == self.parallel_interactions
                parallel = np.asarray(list(range(num_parallel)))

        elif util.is_iterable(x=parallel):
            parallel = np.asarray(parallel)
            num_parallel = parallel.shape[0]
            if reward == 0.0:
                reward = np.asarray([0.0 for _ in range(num_parallel)])
            if terminal is False:
                terminal = np.asarray([0 for _ in range(num_parallel)])

        else:
            reward = np.asarray([float(reward)])
            terminal = np.asarray([int(terminal)])
            parallel = np.asarray([int(parallel)])
            num_parallel = 1

        # Check whether shapes/lengths are consistent
        if parallel.shape[0] == 0:
            raise TensorforceError.value(
                name='Agent.observe', argument='len(parallel)', value=parallel.shape[0], hint='= 0'
            )
        if reward.shape != parallel.shape:
            raise TensorforceError.value(
                name='Agent.observe', argument='len(reward)', value=reward.shape,
                hint='!= parallel length'
            )
        if terminal.shape != parallel.shape:
            raise TensorforceError.value(
                name='Agent.observe', argument='len(terminal)', value=terminal.shape,
                hint='!= parallel length'
            )

        # Check whether current timesteps are not completed
        if self.timestep_completed[parallel].any():
            raise TensorforceError(message="Calling agent.observe must be preceded by agent.act.")
        self.timestep_completed[parallel] = True

        # Process per parallel interaction
        num_updates = 0
        for n in range(num_parallel):

            # Buffer inputs
            p = parallel[n]
            index = self.buffer_indices[p]
            self.buffers['terminal'][p, index] = terminal[n]
            self.buffers['reward'][p, index] = reward[n]

            # Increment buffer index
            index += 1
            self.buffer_indices[p] = index

            # Check whether episode is too long
            if self.max_episode_timesteps is not None and index > self.max_episode_timesteps:
                raise TensorforceError(message="Episode longer than max_episode_timesteps.")

            # Continue if not terminal and buffer_observe
            if terminal == 0 and index < self.config.buffer_observe:
                continue

            # Reset buffer index
            self.buffer_indices[p] = 0

            # Buffered terminal/reward inputs
            terminal = self.buffers['terminal'][p, :index]
            reward = self.buffers['reward'][p, :index]

            # Recorder
            if self.recorder_spec is not None and \
                    self.episodes >= self.recorder_spec.get('start', 0):

                # Store buffered values
                for name in self.states_spec:
                    self.recorded[name].append(self.states_buffers[name][p, :index].copy())
                for name in self.auxiliaries_spec:
                    self.recorded[name].append(self.states_buffers[name][p, :index].copy())
                for name, spec in self.actions_spec.items():
                    self.recorded[name].append(self.actions_buffers[name][p, :index].copy())
                self.recorded['terminal'].append(terminal.copy())
                self.recorded['reward'].append(reward.copy())

                # If terminal
                if terminal[-1] > 0:
                    self.num_episodes += 1

                    # Check whether recording step
                    if self.num_episodes == self.recorder_spec.get('frequency', 1):
                        self.num_episodes = 0

                        # Manage recorder directory
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

                        # Write recording file
                        filename = os.path.join(directory, 'trace-{}-{}.npz'.format(
                            self.episodes, time.strftime('%Y%m%d-%H%M%S')
                        ))
                        np.savez_compressed(
                            file=filename,
                            **util.fmap(function=np.concatenate, xs=self.recorded, depth=1)
                        )

                        # Clear recorded values
                        for recorded in self.recorded.values():
                            recorded.clear()

            # Inputs to tensors
            terminal = self.terminal_spec.to_tensor(value=terminal, batched=True)
            reward = self.reward_spec.to_tensor(value=reward, batched=True)
            parallel_tensor = self.parallel_spec.to_tensor(value=p, batched=False)

            # Model.observe()
            is_updated, self.episodes, self.updates = self.model.observe(
                terminal=terminal, reward=reward, parallel=parallel_tensor
            )
            num_updates += int(is_updated)

        return num_updates

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
            elif filename.endswith('.hdf5') or filename.endswith('.h5'):
                filename = filename[:-5]
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


        def function(name, spec):
            auxiliary = ArrayDict()
            if self.config.enable_int_action_masking and spec.type == 'int' and \
                    spec.num_values is not None:
                # Mask, either part of states or default all true
                auxiliary['mask'] = states.pop(name + '_mask', np.ones(
                    shape=(num_parallel,) + spec.shape + (spec.num_values,), dtype=spec.np_type()
                ))
            return auxiliary

        auxiliaries = self.actions_spec.fmap(function=function, cls=ArrayDict, with_names=True)


        # Normalize states dictionary
        states = util.input2tensor(value=states, spec=self.states_spec)

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
