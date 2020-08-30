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

from collections import OrderedDict
import importlib
import json
import os
import random

import numpy as np
import tensorflow as tf

from tensorforce import util, TensorforceError
from tensorforce.agents import Recorder
import tensorforce.agents
from tensorforce.core import ArrayDict, TensorDict, TensorSpec, TensorforceConfig


class Agent(Recorder):
    """
    Tensorforce agent interface.
    """

    @staticmethod
    def create(agent='tensorforce', environment=None, **kwargs):
        """
        Creates an agent from a specification.

        Args:
            agent (specification | Agent class/object | lambda[states -> actions]): JSON file,
                specification key, configuration dictionary, library module, or `Agent`
                class/object. Alternatively, an act-function mapping states to actions which is
                supposed to be recorded.
                (<span style="color:#00C000"><b>default</b></span>: Tensorforce base agent).
            environment (Environment object): Environment which the agent is supposed to be trained
                on, environment-related arguments like state/action space specifications and
                maximum episode length will be extract if given
                (<span style="color:#C00000"><b>recommended</b></span>).
            kwargs: Additional agent arguments.
        """
        if isinstance(agent, Recorder):
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

        elif (isinstance(agent, type) and issubclass(agent, Agent)) or callable(agent):
            # Type specification, or Recorder
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

            if isinstance(agent, type) and issubclass(agent, Agent):
                agent = agent(**kwargs)
                assert isinstance(agent, Agent)
            else:
                if 'recorder' not in kwargs:
                    raise TensorforceError.required(name='Recorder', argument='recorder')
                agent = Recorder(fn_act=agent, **kwargs)
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

        # Check whether spec attribute exists
        if not hasattr(self, 'spec'):
            raise TensorforceError.required_attribute(name='Agent', attribute='spec')

        # Tensorforce config
        if config is None:
            config = dict()
        self.config = TensorforceConfig(**config)

        # TensorFlow logging
        tf.get_logger().setLevel(self.config.tf_log_level)

        # TensorFlow eager mode
        if self.config.eager_mode:
            tf.config.run_functions_eagerly(run_eagerly=True)

        # Random seed
        if self.config.seed is not None:
            random.seed(a=self.config.seed)
            np.random.seed(seed=self.config.seed)
            tf.random.set_seed(seed=self.config.seed)

        super().__init__(
            fn_act=None, states=states, actions=actions,
            max_episode_timesteps=max_episode_timesteps,
            parallel_interactions=parallel_interactions, recorder=recorder
        )

    def __str__(self):
        return self.__class__.__name__

    def initialize(self):
        """
        Initialize the agent. Automatically triggered as part of Agent.create/load.
        """
        super().initialize()

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
        self.deterministic_spec = self.model.deterministic_spec

        # Parallel observe buffers
        self.terminal_buffer = [list() for _ in range(self.parallel_interactions)]
        self.reward_buffer = [list() for _ in range(self.parallel_interactions)]

        # Store agent spec as JSON
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
        super().close()
        self.model.close()
        del self.model

    def reset(self):
        """
        Resets possibly inconsistent internal values, for instance, after saving and restoring an
        agent. Automatically triggered as part of Agent.create/load/initialize/restore.
        """
        super().reset()

        # Reset observe buffers
        for buffer in self.terminal_buffer:
            buffer.clear()
        for buffer in self.reward_buffer:
            buffer.clear()

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
        return self.model.initial_internals.to_dict()

    def act(
        self, states, internals=None, parallel=0, independent=False, deterministic=False,
        # Deprecated
        evaluation=None
    ):
        """
        Returns action(s) for the given state(s), needs to be followed by `observe()` unless
        independent mode.

        See the [act-observe script](https://github.com/tensorforce/tensorforce/blob/master/examples/act_observe_interface.py)
        for an example application as part of the act-observe interface.

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
                and this call is thus not followed by observe()
                (<span style="color:#00C000"><b>default</b></span>: false).
            deterministic (bool): Whether action should be chosen deterministically, so no
                sampling and no exploration, only valid in independent mode
                (<span style="color:#00C000"><b>default</b></span>: false).

        Returns:
            dict[action] | iter[dict[action]], dict[internal] | iter[dict[internal]] if `internals`
            argument given: Dictionary containing action(s), dictionary containing next internal
            agent state(s) if independent mode.
        """
        if evaluation is not None:
            raise TensorforceError.deprecated(
                name='Agent.act', argument='evaluation', replacement='independent'
            )

        return super().act(
            states=states, internals=internals, parallel=parallel, independent=independent,
            deterministic=deterministic
        )

    def fn_act(
        self, states, internals, parallel, independent, deterministic, is_internals_none,
        num_parallel
    ):

        # Separate auxiliaries
        def function(name, spec):
            auxiliary = ArrayDict()
            if self.config.enable_int_action_masking and spec.type == 'int' and \
                    spec.num_values is not None:
                if name is None:
                    name = 'action'
                # Mask, either part of states or default all true
                auxiliary['mask'] = states.pop(name + '_mask', np.ones(
                    shape=(num_parallel,) + spec.shape + (spec.num_values,), dtype=spec.np_type()
                ))
            return auxiliary

        auxiliaries = self.actions_spec.fmap(function=function, cls=ArrayDict, with_names=True)
        if self.states_spec.is_singleton() and not states.is_singleton():
            states[None] = states.pop('state')

        # Inputs to tensors
        states = self.states_spec.to_tensor(value=states, batched=True)
        if independent and not is_internals_none:
            internals = self.internals_spec.to_tensor(value=internals, batched=True)
        auxiliaries = self.auxiliaries_spec.to_tensor(value=auxiliaries, batched=True)
        if independent:
            deterministic = self.deterministic_spec.to_tensor(value=deterministic, batched=False)

        # Model.act()
        if not independent:
            parallel = self.parallel_spec.to_tensor(value=parallel, batched=True)
            actions, timesteps = self.model.act(
                states=states, auxiliaries=auxiliaries, parallel=parallel
            )
            self.timesteps = timesteps.numpy().item()

        elif len(self.internals_spec) > 0:
            if len(self.auxiliaries_spec) > 0:
                actions, internals = self.model.independent_act(
                    states=states, internals=internals, auxiliaries=auxiliaries,
                    deterministic=deterministic
                )
            else:
                assert len(auxiliaries) == 0
                actions, internals = self.model.independent_act(
                    states=states, internals=internals, deterministic=deterministic
                )

        else:
            if len(self.auxiliaries_spec) > 0:
                actions = self.model.independent_act(
                    states=states, auxiliaries=auxiliaries, deterministic=deterministic
                )
            else:
                assert len(auxiliaries) == 0
                actions = self.model.independent_act(states=states, deterministic=deterministic)

        # Outputs from tensors
        actions = self.actions_spec.from_tensor(tensor=actions, batched=True)
        if independent and len(self.internals_spec) > 0:
            internals = self.internals_spec.from_tensor(tensor=internals, batched=True)

        if self.model.saver is not None:
            self.model.save()

        return actions, internals

    def observe(self, reward=0.0, terminal=False, parallel=0):
        """
        Observes reward and whether a terminal state is reached, needs to be preceded by `act()`.

        See the [act-observe script](https://github.com/tensorforce/tensorforce/blob/master/examples/act_observe_interface.py)
        for an example application as part of the act-observe interface.

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
        reward, terminal, parallel = super().observe(
            reward=reward, terminal=terminal, parallel=parallel
        )

        # Process per parallel interaction
        num_updates = 0
        for p, t, r in zip(parallel.tolist(), terminal.tolist(), reward.tolist()):

            # Buffer inputs
            self.terminal_buffer[p].append(t)
            self.reward_buffer[p].append(r)

            # Continue if not terminal and buffer_observe
            if t == 0 and (
                self.config.buffer_observe == 'episode' or
                len(self.terminal_buffer[p]) < self.config.buffer_observe
            ):
                continue

            # Buffered terminal/reward inputs
            ts = np.asarray(self.terminal_buffer[p], dtype=self.terminal_spec.np_type())
            rs = np.asarray(self.reward_buffer[p], dtype=self.reward_spec.np_type())
            self.terminal_buffer[p].clear()
            self.reward_buffer[p].clear()

            # Inputs to tensors
            terminal_tensor = self.terminal_spec.to_tensor(value=ts, batched=True)
            reward_tensor = self.reward_spec.to_tensor(value=rs, batched=True)
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

    # def get_variables(self):
    #     """
    #     Returns the names of all agent variables.

    #     Returns:
    #         list[str]: Names of variables.
    #     """
    #     return [
    #         variable.name[len(self.model.name) + 1: -2] for variable in self.model.get_variables()
    #     ]

    # def get_variable(self, variable):
    #     """
    #     Returns the value of the variable with the given name.

    #     Args:
    #         variable (string): Variable name
    #             (<span style="color:#C00000"><b>required</b></span>).

    #     Returns:
    #         numpy-array: Variable value.
    #     """
    #     return self.model.get_variable(variable=variable)

    # def assign_variable(self, variable, value):
    #     """
    #     Assigns the given value to the variable with the given name.

    #     Args:
    #         variable (string): Variable name
    #             (<span style="color:#C00000"><b>required</b></span>).
    #         value (variable-compatible value): Value to assign to variable
    #             (<span style="color:#C00000"><b>required</b></span>).
    #     """
    #     self.model.assign_variable(variable=variable, value=value)

    # def summarize(self, summary, value, step=None):
    #     """
    #     Records a value for the given custom summary label (as specified via summarizer[custom]).

    #     Args:
    #         variable (string): Custom summary label
    #             (<span style="color:#C00000"><b>required</b></span>).
    #         value (summary-compatible value): Summary value to record
    #             (<span style="color:#C00000"><b>required</b></span>).
    #         step (int): Summary recording step
    #             (<span style="color:#00C000"><b>default</b></span>: current timestep).
    #     """
    #     self.model.summarize(summary=summary, value=value, step=step)

    # def get_output_tensors(self, function):
    #     """
    #     Returns the names of output tensors for the given function.

    #     Args:
    #         function (str): Function name
    #             (<span style="color:#C00000"><b>required</b></span>).

    #     Returns:
    #         list[str]: Names of output tensors.
    #     """
    #     if function in self.model.output_tensors:
    #         return self.model.output_tensors[function]
    #     else:
    #         raise TensorforceError.value(
    #             name='agent.get_output_tensors', argument='function', value=function
    #         )

    # def get_available_summaries(self):
    #     """
    #     Returns the summary labels provided by the agent.

    #     Returns:
    #         list[str]: Available summary labels.
    #     """
    #     return self.model.get_available_summaries()

    # def get_query_tensors(self, function):
    #     """
    #     Returns the names of queryable tensors for the given function.

    #     Args:
    #         function (str): Function name
    #             (<span style="color:#C00000"><b>required</b></span>).

    #     Returns:
    #         list[str]: Names of queryable tensors.
    #     """
    #     if function in self.model.query_tensors:
    #         return self.model.query_tensors[function]
    #     else:
    #         raise TensorforceError.value(
    #             name='agent.get_query_tensors', argument='function', value=function
    #         )


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
