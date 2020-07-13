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
from tensorforce.core import ArrayDict, ListDict, TensorDict, TensorforceConfig, TensorSpec


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
                (<span style="color:#C00000"><b>recommended</b></span>).
            kwargs: Additional agent arguments.
        """
        if isinstance(agent, Agent):
            if environment is not None:
                # TODO:
                # assert agent.spec['states'] == environment.states()
                # assert agent.spec['actions'] == environment.actions()
                # assert environment.max_episode_timesteps() is None or \
                #     agent.spec['max_episode_timesteps'] >= environment.max_episode_timesteps()
                pass

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
                    # TODO:
                    # assert kwargs['states'] == environment.states()
                    pass
                else:
                    kwargs['states'] = environment.states()
                if 'actions' in kwargs:
                    # assert kwargs['actions'] == environment.actions()
                    pass
                else:
                    kwargs['actions'] = environment.actions()
                if environment.max_episode_timesteps() is None:
                    pass
                elif 'max_episode_timesteps' in kwargs:
                    # assert kwargs['max_episode_timesteps'] >= environment.max_episode_timesteps()
                    pass
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
                raise TensorforceError.value(name='Agent.create', argument='agent', value=agent)

        else:
            raise TensorforceError.type(name='Agent.create', argument='agent', dtype=type(agent))

    @staticmethod
    def load(directory=None, filename=None, format=None, environment=None, **kwargs):
        """
        Restores an agent from a directory/file.

        Args:
            directory (str): Checkpoint directory
                (<span style="color:#C00000"><b>required</b></span>, unless saver is specified).
            filename (str): Checkpoint filename, with or without append and extension
                (<span style="color:#00C000"><b>default</b></span>: "agent").
            format ("checkpoint" | "saved-model" | "numpy" | "hdf5"): File format, "saved-model" loads
                an act-only agent based on a Protobuf model
                (<span style="color:#00C000"><b>default</b></span>: format matching directory and
                filename, required to be unambiguous).
            environment (Environment object): Environment which the agent is supposed to be trained
                on, environment-related arguments like state/action space specifications and
                maximum episode length will be extract if given
                (<span style="color:#C00000"><b>recommended</b></span>).
            kwargs: Additional agent arguments.
        """
        if directory is not None:
            if filename is None:
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

        agent.pop('internals', None)
        agent.pop('initial_internals', None)
        saver_restore = False
        if 'saver' in agent and isinstance(agent['saver'], dict):
            if not agent.get('load', True):
                raise TensorforceError.value(
                    name='Agent.load', argument='saver[load]', value=agent['saver']['load']
                )
            agent['saver'] = dict(agent['saver'])
            agent['saver']['load'] = True
            saver_restore = True
        elif 'saver' in kwargs and isinstance(kwargs['saver'], dict):
            if not kwargs.get('load', True):
                raise TensorforceError.value(
                    name='Agent.load', argument='saver[load]', value=kwargs['saver']['load']
                )
            kwargs['saver'] = dict(kwargs['saver'])
            kwargs['saver']['load'] = True
            saver_restore = True
        agent = Agent.create(agent=agent, environment=environment, **kwargs)
        if not saver_restore:
            agent.restore(directory=directory, filename=filename, format=format)

        return agent

    def __init__(
        self, states, actions, max_episode_timesteps=None, parallel_interactions=1, config=None,
        recorder=None
    ):
        util.overwrite_staticmethod(obj=self, function='create')
        util.overwrite_staticmethod(obj=self, function='load')

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

        # Config
        if config is None:
            config = dict()
        self.config = TensorforceConfig(**config)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.config.tf_log_level)

        # Import tensorflow after setting log level
        import tensorflow as tf

        # Eager mode
        if self.config.eager_mode:
            tf.config.experimental_run_functions_eagerly(run_eagerly=True)

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
        Initialize the agent. Automatically triggered as part of Agent.create/load.
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
        self.model.initialize()

        # Value space specifications
        self.states_spec = self.model.states_spec
        self.internals_spec = self.model.internals_spec
        self.auxiliaries_spec = self.model.auxiliaries_spec
        self.actions_spec = self.model.actions_spec
        self.terminal_spec = self.model.terminal_spec
        self.reward_spec = self.model.reward_spec
        self.parallel_spec = self.model.parallel_spec

        # Act-observe timestep completed check
        self.timestep_completed = np.ones(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='bool')
        )

        # Parallel terminal/reward buffers
        self.buffers = ListDict()
        self.buffers['terminal'] = [list() for _ in range(self.parallel_interactions)]
        self.buffers['reward'] = [list() for _ in range(self.parallel_interactions)]

        # Recorder buffers if required
        if self.recorder_spec is not None:
            self.num_episodes = 0

            def function(spec):
                return [list() for _ in range(self.parallel_interactions)]

            self.buffers['states'] = self.states_spec.fmap(function=function, cls=ListDict)
            self.buffers['auxiliaries'] = self.auxiliaries_spec.fmap(
                function=function, cls=ListDict
            )
            self.buffers['actions'] = self.actions_spec.fmap(function=function, cls=ListDict)

            function = (lambda x: list())

            self.recorded = ListDict()
            self.recorded['states'] = self.states_spec.fmap(function=function, cls=ListDict)
            self.recorded['auxiliaries'] = self.auxiliaries_spec.fmap(
                function=function, cls=ListDict
            )
            self.recorded['actions'] = self.actions_spec.fmap(function=function, cls=ListDict)
            self.recorded['terminal'] = list()
            self.recorded['reward'] = list()

        if self.model.saver is not None:
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
        timesteps, episodes, updates = self.model.reset()
        self.timesteps = timesteps.numpy().item()
        self.episodes = episodes.numpy().item()
        self.updates = updates.numpy().item()

    def close(self):
        """
        Closes the agent.
        """
        self.model.close()
        del self.model

    def reset(self):
        """
        Resets possibly inconsistent internal values, for instance, after saving and restoring an
        agent. Automatically triggered as part of Agent.create/load/initialize/restore.
        """
        # Reset timestep completed
        self.timestep_completed = np.ones(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='bool')
        )

        # Reset buffers
        for buffer in self.buffers.values():
            for x in buffer:
                x.clear()
        if self.recorder_spec is not None:
            for x in self.recorded.values():
                x.clear()

        # Reset model
        timesteps, episodes, updates = self.model.reset()
        self.timesteps = timesteps.numpy().item()
        self.episodes = episodes.numpy().item()
        self.updates = updates.numpy().item()

        if self.model.saver is not None:
            self.model.save()

    def initial_internals(self):
        """
        Returns the initial internal agent state(s), to be used at the beginning of an episode as
        `internals` argument for `act()` in independent mode

        Returns:
            dict[internal]: Dictionary containing initial internal agent state(s).
        """
        return self.model.internals_init.fmap(function=(lambda x: x), cls=OrderedDict)

    def act(
        self, states, internals=None, parallel=0, independent=False,
        # Deprecated
        deterministic=None, evaluation=None
    ):
        """
        Returns action(s) for the given state(s), needs to be followed by `observe()` unless
        independent mode.

        Args:
            states (dict[state] | iter[dict[state]]): Dictionary containing state(s) to be acted on
                (<span style="color:#C00000"><b>required</b></span>).
            internals (dict[internal] | iter[dict[internal]]): Dictionary containing current
                internal agent state(s), either given by `initial_internals()` at the beginning of
                an episode or as return value of the preceding `act()` call
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
        if deterministic is not None:
            raise TensorforceError.deprecated(
                name='Agent.act', argument='deterministic', replacement='independent'
            )
        if evaluation is not None:
            raise TensorforceError.deprecated(
                name='Agent.act', argument='evaluation', replacement='independent'
            )

        # Independent and internals
        if independent:
            if parallel != 0:
                raise TensorforceError.invalid(
                    name='Agent.act', argument='parallel', condition='independent is true'
                )
            is_internals_none = (internals is None)
            if is_internals_none and len(self.internals_spec) > 0:
                raise TensorforceError.required(
                    name='Agent.act', argument='internals', condition='independent is true'
                )
        else:
            if internals is not None:
                raise TensorforceError.invalid(
                    name='Agent.act', argument='internals', condition='independent is false'
                )

        # Process states input and infer batching structure
        states, batched, num_parallel, is_iter_of_dicts, input_type = self._process_states_input(
            states=states, function_name='Agent.act'
        )

        if independent:
            # Independent mode: handle internals argument

            if is_internals_none:
                # Default input internals=None
                pass

            elif is_iter_of_dicts:
                # Input structure iter[dict[internal]]
                if not isinstance(internals, (tuple, list)):
                    raise TensorforceError.type(
                        name='Agent.act', argument='internals', dtype=type(internals),
                        hint='is not tuple/list'
                    )
                internals = [ArrayDict(internal) for internal in internals]
                internals = internals[0].fmap(
                    function=(lambda *xs: np.stack(xs, axis=0)), zip_values=internals[1:]
                )

            else:
                # Input structure dict[iter[internal]]
                if not isinstance(internals, dict):
                    raise TensorforceError.type(
                        name='Agent.act', argument='internals', dtype=type(internals),
                        hint='is not dict'
                    )
                internals = ArrayDict(internals)

            if not independent or not is_internals_none:
                # Expand inputs if not batched
                if not batched:
                    internals = internals.fmap(function=(lambda x: np.expand_dims(x, axis=0)))

                # Check number of inputs
                for name, internal in internals.items():
                    if internal.shape[0] != num_parallel:
                        raise TensorforceError.value(
                            name='Agent.act', argument='len(internals[{}])'.format(name),
                            value=internal.shape[0], hint='!= len(states)'
                        )

        else:
            # Non-independent mode: handle parallel input

            if parallel == 0:
                # Default input parallel=0
                if batched:
                    assert num_parallel == self.parallel_interactions
                    parallel = np.asarray(list(range(num_parallel)))
                else:
                    parallel = np.asarray([parallel])

            elif batched:
                # Batched input
                parallel = np.asarray(parallel)

            else:
                # Expand input if not batched
                parallel = np.asarray([parallel])

            # Check number of inputs
            if parallel.shape[0] != num_parallel:
                raise TensorforceError.value(
                    name='Agent.act', argument='len(parallel)', value=len(parallel),
                    hint='!= len(states)'
                )

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

        # If not independent, check whether previous timesteps were completed
        if not independent:
            if not self.timestep_completed[parallel].all():
                raise TensorforceError(
                    message="Calling agent.act must be preceded by agent.observe."
                )
            self.timestep_completed[parallel] = False

        # Buffer inputs for recording
        if self.recorder_spec is not None and not independent and \
                self.episodes >= self.recorder_spec.get('start', 0):
            for n in range(num_parallel):
                for name in self.states_spec:
                    self.buffers['states'][name][parallel[n]].append(states[name][n])
                for name in self.auxiliaries_spec:
                    self.buffers['auxiliaries'][name][parallel[n]].append(auxiliaries[name][n])

        # Inputs to tensors
        states = self.states_spec.to_tensor(value=states, batched=True)
        if independent and not is_internals_none:
            internals = self.internals_spec.to_tensor(value=internals, batched=True)
        auxiliaries = self.auxiliaries_spec.to_tensor(value=auxiliaries, batched=True)
        parallel_tensor = self.parallel_spec.to_tensor(value=parallel, batched=True)

        # Model.act()
        if not independent:
            actions, timesteps = self.model.act(
                states=states, auxiliaries=auxiliaries, parallel=parallel_tensor
            )
            self.timesteps = timesteps.numpy().item()

        elif len(self.internals_spec) > 0:
            if len(self.auxiliaries_spec) > 0:
                actions_internals = self.model.independent_act(
                    states=states, internals=internals, auxiliaries=auxiliaries
                )
            else:
                assert len(auxiliaries) == 0
                actions_internals = self.model.independent_act(states=states, internals=internals)
            actions_internals = TensorDict(actions_internals)
            actions = actions_internals['actions']
            internals = actions_internals['internals']

        else:
            if len(self.auxiliaries_spec) > 0:
                actions = self.model.independent_act(states=states, auxiliaries=auxiliaries)
            else:
                assert len(auxiliaries) == 0
                actions = self.model.independent_act(states=states)
            actions = TensorDict(actions)

        # Outputs from tensors
        # print(actions)
        actions = self.actions_spec.from_tensor(tensor=actions, batched=True)

        # Buffer outputs for recording
        if self.recorder_spec is not None and not independent and \
                self.episodes >= self.recorder_spec.get('start', 0):
            for n in range(num_parallel):
                for name in self.actions_spec:
                    self.buffers['actions'][name][parallel[n]].append(actions[name][n])

        # Unbatch actions
        if batched:
            # If inputs were batched, turn list of dicts into dict of lists
            function = (lambda x: x.item() if x.shape == () else x)
            if self.single_action:
                actions = input_type(function(actions['action'][n]) for n in range(num_parallel))
            else:
                # TODO: recursive
                actions = input_type(
                    OrderedDict(((name, function(x[n])) for name, x in actions.items()))
                    for n in range(num_parallel)
                )

            if independent and not is_internals_none and is_iter_of_dicts:
                # TODO: recursive
                internals = input_type(
                    OrderedDict(((name, function(x[n])) for name, x in internals.items()))
                    for n in range(num_parallel)
                )

        else:
            # If inputs were not batched, unbatch outputs
            function = (lambda x: x.item() if x.shape == (1,) else x[0])
            if self.single_action:
                actions = function(actions['action'])
            else:
                actions = actions.fmap(function=function, cls=OrderedDict)
            if independent and not is_internals_none:
                internals = internals.fmap(function=function, cls=OrderedDict)

        if self.model.saver is not None:
            self.model.save()

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
            else:
                terminal = np.asarray(terminal)
            if parallel == 0:
                assert num_parallel == self.parallel_interactions
                parallel = np.asarray(list(range(num_parallel)))
            else:
                parallel = np.asarray(parallel)

        elif util.is_iterable(x=terminal):
            terminal = np.asarray([int(t) for t in terminal])
            num_parallel = terminal.shape[0]
            if reward == 0.0:
                reward = np.asarray([0.0 for _ in range(num_parallel)])
            else:
                reward = np.asarray(reward)
            if parallel == 0:
                assert num_parallel == self.parallel_interactions
                parallel = np.asarray(list(range(num_parallel)))
            else:
                parallel = np.asarray(parallel)

        elif util.is_iterable(x=parallel):
            parallel = np.asarray(parallel)
            num_parallel = parallel.shape[0]
            if reward == 0.0:
                reward = np.asarray([0.0 for _ in range(num_parallel)])
            else:
                reward = np.asarray(reward)
            if terminal is False:
                terminal = np.asarray([0 for _ in range(num_parallel)])
            else:
                terminal = np.asarray(terminal)

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

        # Convert terminal to int if necessary
        if terminal.dtype is util.np_dtype(dtype='bool'):
            zeros = np.zeros_like(terminal, dtype=util.np_dtype(dtype='int'))
            ones = np.ones_like(terminal, dtype=util.np_dtype(dtype='int'))
            terminal = np.where(terminal, ones, zeros)

        # Check whether current timesteps are not completed
        if self.timestep_completed[parallel].any():
            raise TensorforceError(message="Calling agent.observe must be preceded by agent.act.")
        self.timestep_completed[parallel] = True

        # Process per parallel interaction
        num_updates = 0
        for n in range(num_parallel):

            # Buffer inputs
            p = parallel[n]
            self.buffers['terminal'][p].append(terminal[n])
            self.buffers['reward'][p].append(reward[n])

            # Check whether episode is too long
            if self.max_episode_timesteps is not None and \
                    len(self.buffers['terminal'][p]) > self.max_episode_timesteps:
                raise TensorforceError(message="Episode longer than max_episode_timesteps.")

            # Continue if not terminal and buffer_observe
            if terminal[n].item() == 0 and (
                self.config.buffer_observe == 'episode' or
                len(self.buffers['terminal'][p]) < self.config.buffer_observe
            ):
                continue

            # Buffered terminal/reward inputs
            t = np.asarray(self.buffers['terminal'][p], dtype=self.terminal_spec.np_type())
            r = np.asarray(self.buffers['reward'][p], dtype=self.reward_spec.np_type())
            self.buffers['terminal'][p].clear()
            self.buffers['reward'][p].clear()

            # Recorder
            if self.recorder_spec is not None and \
                    self.episodes >= self.recorder_spec.get('start', 0):

                # Store buffered values
                for name in self.states_spec:
                    self.recorded['states'][name].append(
                        np.stack(self.buffers['states'][name][p], axis=0)
                    )
                    self.buffers['states'][name][p].clear()
                for name in self.auxiliaries_spec:
                    self.recorded['auxiliaries'][name].append(
                        np.stack(self.buffers['auxiliaries'][name][p], axis=0)
                    )
                    self.buffers['auxiliaries'][name][p].clear()
                for name, spec in self.actions_spec.items():
                    self.recorded['actions'][name].append(
                        np.stack(self.buffers['actions'][name][p], axis=0)
                    )
                    self.buffers['actions'][name][p].clear()
                self.recorded['terminal'].append(t.copy())
                self.recorded['reward'].append(r.copy())

                # If terminal
                if t[-1] > 0:
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
                                and os.path.splitext(f)[1] == '.npz'
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
                        filename = os.path.join(directory, 'trace-{}-{:09d}.npz'.format(
                            time.strftime('%Y%m%d-%H%M%S'), self.episodes
                        ))
                        kwargs = self.recorded.fmap(function=np.concatenate, cls=ArrayDict).items()
                        np.savez_compressed(file=filename, **dict(kwargs))

                        # Clear recorded values
                        for recorded in self.recorded.values():
                            recorded.clear()

            # Inputs to tensors
            terminal_tensor = self.terminal_spec.to_tensor(value=t, batched=True)
            reward_tensor = self.reward_spec.to_tensor(value=r, batched=True)
            parallel_tensor = self.parallel_spec.to_tensor(value=p, batched=False)

            # Model.observe()
            updated, episodes, updates = self.model.observe(
                terminal=terminal_tensor, reward=reward_tensor, parallel=parallel_tensor
            )
            num_updates += int(updated.numpy().item())
            self.episodes = episodes.numpy().item()
            self.updates = updates.numpy().item()

        if self.model.saver is not None:
            self.model.save()

        return num_updates

    def _process_states_input(self, states, function_name):
        if self.single_state and not isinstance(states, dict) and not (
            util.is_iterable(x=states) and isinstance(states[0], dict)
        ):
            # Single state
            input_type = type(states)
            states = np.asarray(states)

            if states.shape == self.states_spec['state'].shape:
                # Single state is not batched
                states = ArrayDict(state=np.expand_dims(states, axis=0))
                batched = False
                num_instances = 1
                is_iter_of_dicts = None
                input_type = None

            else:
                # Single state is batched, iter[state]
                assert states.shape[1:] == self.states_spec['state'].shape
                assert input_type in (tuple, list, np.ndarray)
                num_instances = states.shape[0]
                states = ArrayDict(state=states)
                batched = True
                is_iter_of_dicts = True  # Default

        elif util.is_iterable(x=states):
            # States is batched, iter[dict[state]]
            batched = True
            num_instances = len(states)
            is_iter_of_dicts = True
            input_type = type(states)
            assert input_type in (tuple, list)
            if num_instances == 0:
                raise TensorforceError.value(
                    name=function_name, argument='len(states)', value=num_instances, hint='= 0'
                )
            for n, state in enumerate(states):
                if not isinstance(state, dict):
                    raise TensorforceError.type(
                        name=function_name, argument='states[{}]'.format(n), dtype=type(state),
                        hint='is not dict'
                    )
            # Turn iter of dicts into dict of arrays
            # (Doesn't use self.states_spec since states also contains auxiliaries)
            states = [ArrayDict(state) for state in states]
            states = states[0].fmap(
                function=(lambda *xs: np.stack(xs, axis=0)), zip_values=states[1:]
            )

        elif isinstance(states, dict):
            # States is dict, turn into arrays
            some_state = next(iter(states.values()))
            input_type = type(some_state)

            states = ArrayDict(states)

            name, spec = self.states_spec.item()
            if states[name].shape == spec.shape:
                # States is not batched, dict[state]
                states = states.fmap(function=(lambda state: np.expand_dims(state, axis=0)))
                batched = False
                num_instances = 1
                is_iter_of_dicts = None
                input_type = None

            else:
                # States is batched, dict[iter[state]]
                assert states[name].shape[1:] == spec.shape
                assert input_type in (tuple, list, np.ndarray)
                batched = True
                num_instances = states[name].shape[0]
                is_iter_of_dicts = False
                if num_instances == 0:
                    raise TensorforceError.value(
                        name=function_name, argument='len(states)', value=num_instances, hint='= 0'
                    )

        else:
            raise TensorforceError.type(
                name=function_name, argument='states', dtype=type(states),
                hint='is not array/tuple/list/dict'
            )

        # Check number of inputs
        if any(state.shape[0] != num_instances for state in states.values()):
            raise TensorforceError.value(
                name=function_name, argument='len(states)',
                value=[state.shape[0] for state in states.values()], hint='inconsistent'
            )

        return states, batched, num_instances, is_iter_of_dicts, input_type

    def save(self, directory, filename=None, format='checkpoint', append=None):
        """
        Saves the agent to a checkpoint.

        Args:
            directory (str): Checkpoint directory
                (<span style="color:#C00000"><b>required</b></span>).
            filename (str): Checkpoint filename, without extension
                (<span style="color:#C00000"><b>required</b></span>, unless "saved-model" format).
            format ("checkpoint" | "saved-model" | "numpy" | "hdf5"): File format, "checkpoint" uses
                TensorFlow Checkpoint to save model, "saved-model" uses TensorFlow SavedModel to
                save an optimized act-only model, whereas the others store only variables as
                NumPy/HDF5 file
                (<span style="color:#00C000"><b>default</b></span>: TensorFlow Checkpoint).
            append ("timesteps" | "episodes" | "updates"): Append timestep/episode/update to
                checkpoint filename
                (<span style="color:#00C000"><b>default</b></span>: none).

        Returns:
            str: Checkpoint path.
        """
        # TODO: Messes with required parallel disentangling, better to remove unfinished episodes
        # from memory, but currently entire episode buffered anyway...
        # Empty buffers before saving
        # for parallel in range(self.parallel_interactions):
        #     if self.buffer_indices[parallel] > 0:
        #         self.model_observe(parallel=parallel)

        path = self.model.save(directory=directory, filename=filename, format=format, append=append)

        if filename is None:
            filename = self.model.name
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
                (<span style="color:#C00000"><b>required</b></span>, unless "saved-model" format and
                saver specified).
            filename (str): Checkpoint filename, with or without append and extension
                (<span style="color:#C00000"><b>required</b></span>, unless "saved-model" format and
                saver specified).
            format ("checkpoint" | "numpy" | "hdf5"): File format
                (<span style="color:#00C000"><b>default</b></span>: format matching directory and
                filename, required to be unambiguous).
        """
        if not hasattr(self, 'model'):
            raise TensorforceError(message="Missing agent attribute model.")

        if not self.is_initialized:
            self.initialize()

        # format implicitly given if file exists
        if format is None and os.path.isfile(os.path.join(directory, filename)):
            if '.data-' in filename:
                filename = filename[:filename.index('.data-')]
                format = 'checkpoint'
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
        elif format is None and os.path.isfile(os.path.join(directory, filename + '.index')):
            format = 'checkpoint'
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
                        (name == filename + '.hdf5' or name == filename + '.h5'):
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
                    format = 'checkpoint'
                else:
                    assert format == 'checkpoint'
                if filename is None or \
                        not os.path.isfile(os.path.join(directory, filename + '.index')):
                    import tensorflow as tf
                    path = tf.train.latest_checkpoint(checkpoint_dir=directory)
                    if not path:
                        raise TensorforceError.exists_not(name='Checkpoint', value=directory)
                    filename = os.path.basename(path)

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
        elif isinstance(obj, TensorSpec):
            return obj.json()
        else:
            return super().default(obj)
