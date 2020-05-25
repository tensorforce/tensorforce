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

import os
import time

import h5py
import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import ArrayDict, Module, ModuleDict, parameter_modules, SignatureDict, \
    TensorDict, TensorSpec, TensorsSpec, tf_function, tf_util, VariableDict
from tensorforce.core.networks import Preprocessor


class Model(Module):

    def __init__(
        self, *,
        # Model
        states, actions, preprocessing, exploration, variable_noise, l2_regularization, name,
        device, parallel_interactions, config, saver, summarizer
    ):
        # Tensorforce config (TODO: should be part of Module init?)
        self.config = config

        Module._MODULE_STACK.clear()
        Module._MODULE_STACK.append(self.__class__)

        super().__init__(device=device, l2_regularization=l2_regularization, name=name)

        assert self.l2_regularization is not None
        self.is_trainable = True
        self.is_saved = True

        # Keep track of tensor names to check for collisions
        self.value_names = set()

        # Terminal specification
        self.terminal_spec = TensorSpec(type='int', shape=(), num_values=3)
        self.value_names.add('terminal')

        # Reward specification
        self.reward_spec = TensorSpec(type='float', shape=())
        self.value_names.add('reward')

        # Parallel specification
        self.parallel_spec = TensorSpec(type='int', shape=(), num_values=parallel_interactions)
        self.value_names.add('parallel')

        # State space specification
        self.unprocessed_states_spec = TensorsSpec(states)
        self.states_spec = TensorsSpec()  # below

        # Check for name collisions
        for name in self.unprocessed_states_spec:
            if name in self.value_names:
                raise TensorforceError.exists(name='value name', value=name)
            self.value_names.add(name)

        # Action space specification
        self.actions_spec = TensorsSpec(actions)

        # Check for name collisions
        for name in self.actions_spec:
            if name in self.value_names:
                raise TensorforceError.exists(name='value name', value=name)
            self.value_names.add(name)

        # Internal state space specification
        self.internals_spec = TensorsSpec()
        self.internals_init = ArrayDict()

        # Auxiliary value space specification
        self.auxiliaries_spec = TensorsSpec()
        for name, spec in self.actions_spec.items():
            if self.config.enable_int_action_masking and spec.type == 'int' and \
                    spec.num_values is not None:
                self.auxiliaries_spec[name] = TensorsSpec(mask=TensorSpec(
                    type='bool', shape=(spec.shape + (spec.num_values,))
                ))

        # States preprocessing
        self.preprocessing = ModuleDict()
        for name, spec in self.unprocessed_states_spec.items():
            if preprocessing is None:
                layers = None
            elif name in preprocessing:
                # TODO: recursive lookup
                layers = preprocessing[name]
            elif spec.type in preprocessing:
                layers = preprocessing[spec.type]
            else:
                layers = None
            if layers is None:
                self.states_spec[name] = self.unprocessed_states_spec[name]
            else:
                self.preprocessing[name] = self.submodule(
                    name=(name + '_preprocessing'), module=Preprocessor, is_trainable=False,
                    input_spec=spec, layers=layers
                )
                self.states_spec[name] = self.preprocessing[name].output_spec()

        # Reward preprocessing
        if preprocessing is not None:
            if 'reward' in preprocessing:
                self.preprocessing['reward'] = self.submodule(
                    name=('reward_preprocessing'), module=Preprocessor, is_trainable=False,
                    input_spec=self.reward_spec, layers=preprocessing['reward']
                )
                if self.preprocessing['reward'].output_spec() != self.reward_spec:
                    raise TensorforceError.mismatch(
                        name='preprocessing', argument='reward output spec',
                        value1=self.preprocessing['reward'].output_spec(),
                        value2=self.reward_spec
                    )

        # Exploration
        # TODO: recursive lookup?
        if isinstance(exploration, dict) and all(name in self.actions_spec for name in exploration):
            # Different exploration per action
            self.exploration = ModuleDict()
            for name, spec in self.actions_spec.items():
                # TODO: recursive lookup
                if name in exploration:
                    # TODO: recursive lookup
                    module = exploration[name]
                elif spec.type in exploration:
                    module = exploration[spec.type]
                else:
                    module = None
                if module is None:
                    pass
                elif spec.type in ('bool', 'int'):
                    self.exploration[name] = self.submodule(
                        name=(name + '_exploration'), module=module, modules=parameter_modules,
                        is_trainable=False, dtype='float', min_value=0.0, max_value=1.0
                    )
                else:
                    self.exploration[name] = self.submodule(
                        name=(name + '_exploration'), module=module, modules=parameter_modules,
                        is_trainable=False, dtype='float', min_value=0.0
                    )
        else:
            # Same exploration for all actions
            self.exploration = self.submodule(
                name='exploration', module=exploration, modules=parameter_modules,
                is_trainable=False, dtype='float', min_value=0.0
            )

        # Variable noise
        self.variable_noise = self.submodule(
            name='variable_noise', module=variable_noise, modules=parameter_modules,
            is_trainable=False, dtype='float', min_value=0.0
        )

        # Parallel interactions
        assert isinstance(parallel_interactions, int) and parallel_interactions >= 1
        self.parallel_interactions = parallel_interactions

        # Saver
        if saver is None:
            self.saver = None
        elif not all(key in (
            'directory', 'filename', 'frequency', 'load', 'max-checkpoints', 'max-hour-frequency',
            'unit'
        ) for key in saver):
            raise TensorforceError.value(
                name='agent', argument='saver', value=list(saver),
                hint='not from {directory,filename,frequency,load,max-checkpoints,'
                     'max-hour-frequency,unit}'
            )
        elif 'directory' not in saver:
            raise TensorforceError.required(name='agent', argument='saver[directory]')
        elif 'frequency' not in saver:
            raise TensorforceError.required(name='agent', argument='saver[frequency]')
        else:
            self.saver = dict(saver)

        # Summarizer
        if summarizer is None:
            self.summarizer = None
            self.summary_labels = frozenset()
        elif not all(
            key in ('directory', 'flush', 'labels', 'max-summaries') for key in summarizer
        ):
            raise TensorforceError.value(
                name='agent', argument='summarizer', value=list(summarizer),
                hint='not from {directory,flush,labels,max-summaries}'
            )
        elif 'directory' not in summarizer:
            raise TensorforceError.required(name='agent', argument='summarizer[directory]')
        else:
            self.summarizer = dict(summarizer)

            # Summary labels
            summary_labels = summarizer.get('labels', ('graph',))
            if summary_labels == 'all':
                self.summary_labels = 'all'
            elif not all(isinstance(label, str) for label in summary_labels):
                raise TensorforceError.value(
                    name='agent', argument='summarizer[labels]', value=summary_labels
                )
            else:
                self.summary_labels = frozenset(summary_labels)

    @property
    def root(self):
        return self

    def close(self):
        if self.saver is not None:
            self.save()
        if self.summarizer is not None:
            self.summarizer.close()

    def __enter__(self):
        assert self.is_initialized is not None
        if self.is_initialized:
            Module._MODULE_STACK.append(self)
        else:
            # Hack: keep non-empty module stack from constructor
            assert len(Module._MODULE_STACK) == 1 and Module._MODULE_STACK[0] is self
        self.device.__enter__()
        self.name_scope.__enter__()
        return self

    def __exit__(self, etype, exception, traceback):
        self.name_scope.__exit__(etype, exception, traceback)
        self.device.__exit__(etype, exception, traceback)
        popped = Module._MODULE_STACK.pop()
        assert popped is self
        assert self.is_initialized is not None
        if not self.is_initialized:
            assert len(Module._MODULE_STACK) == 0

    def initialize(self):
        assert self.is_initialized is None
        self.is_initialized = False

        with self:

            if self.summarizer is not None:
                directory = self.summarizer['directory']
                if os.path.isdir(directory):
                    directories = sorted(
                        d for d in os.listdir(directory)
                        if os.path.isdir(os.path.join(directory, d)) and d.startswith('summary-')
                    )
                else:
                    os.makedirs(directory)
                    directories = list()

                max_summaries = self.summarizer.get('max-summaries', 5)
                if len(directories) > max_summaries - 1:
                    for subdir in directories[:len(directories) - max_summaries + 1]:
                        subdir = os.path.join(directory, subdir)
                        os.remove(os.path.join(subdir, os.listdir(subdir)[0]))
                        os.rmdir(subdir)

                logdir = os.path.join(directory, time.strftime('summary-%Y%m%d-%H%M%S'))
                flush_millis = (self.summarizer.get('flush', 10) * 1000)
                # with tf.name_scope(name='summarizer'):
                self.summarizer = tf.summary.create_file_writer(
                    logdir=logdir, max_queue=None, flush_millis=flush_millis, filename_suffix=None,
                    name='summarizer'
                )

                # TODO: write agent spec?
                # tf.summary.text(name, data, step=None, description=None)

            super().initialize()

            self.core_initialize()

            # Units, used in: Parameter, Model.save(), Model.summarizer????
            self.units = dict(
                timesteps=self.timesteps, episodes=self.episodes, updates=self.updates
            )

            # Checkpoint manager
            if self.saver is not None:
                self.saver_directory = self.saver['directory']
                self.saver_filename = self.saver.get('filename', self.name)
                load = self.saver.get('load', False)
                # with tf.name_scope(name='saver'):
                self.checkpoint = tf.train.Checkpoint(**{self.name: self})
                self.saver = tf.train.CheckpointManager(
                    checkpoint=self.checkpoint, directory=self.saver_directory,
                    max_to_keep=self.saver.get('max-checkpoints', 5),
                    keep_checkpoint_every_n_hours=self.saver.get('max-hour-frequency'),
                    checkpoint_name=self.saver_filename,
                    step_counter=self.units[self.saver.get('unit', 'updates')],
                    checkpoint_interval=self.saver['frequency'], init_fn=None
                )

        self.is_initialized = True
        self.initialize_api()

        if self.saver is not None:
            if load:
                self.restore()
            else:
                self.save()

    def core_initialize(self):
        # Timestep counter
        self.timesteps = self.variable(
            name='timesteps', spec=TensorSpec(type='int'), initializer='zeros', is_trainable=False,
            is_saved=True
        )

        # Episode counter
        self.episodes = self.variable(
            name='episodes', spec=TensorSpec(type='int'), initializer='zeros', is_trainable=False,
            is_saved=True
        )

        # Update counter
        self.updates = self.variable(
            name='updates', spec=TensorSpec(type='int'), initializer='zeros', is_trainable=False,
            is_saved=True
        )

        # Episode reward
        self.episode_reward = self.variable(
            name='episode-reward',
            spec=TensorSpec(type=self.reward_spec.type, shape=(self.parallel_interactions,)),
            initializer='zeros', is_trainable=False, is_saved=False
        )

        # Preprocessed episode reward
        if 'reward' in self.preprocessing:
            self.preprocessed_episode_reward = self.variable(
                name='preprocessed-episode-reward',
                spec=TensorSpec(type=self.reward_spec.type, shape=(self.parallel_interactions,)),
                initializer='zeros', is_trainable=False, is_saved=False
            )

        # Buffer index
        self.buffer_index = self.variable(
            name='buffer-index', spec=TensorSpec(type='int', shape=(self.parallel_interactions,)),
            initializer='zeros', is_trainable=False, is_saved=False
        )

        # States/auxiliaries/actions buffers
        function = (lambda name, spec: self.variable(
            name=(name.replace('/', '_') + '-buffer'), spec=TensorSpec(
                type=spec.type,
                shape=((self.parallel_interactions, self.config.buffer_observe) + spec.shape)
            ), initializer='zeros', is_trainable=False, is_saved=False
        ))
        self.states_buffer = self.states_spec.fmap(
            function=function, cls=VariableDict, with_names=True
        )
        self.auxiliaries_buffer = self.auxiliaries_spec.fmap(
            function=function, cls=VariableDict, with_names=True
        )
        self.actions_buffer = self.actions_spec.fmap(
            function=function, cls=VariableDict, with_names=True
        )

        # Internals buffers
        def function(name, spec, initial):
            shape = ((self.parallel_interactions, self.config.buffer_observe + 1) + spec.shape)
            initializer = np.zeros(shape=shape, dtype=util.np_dtype(dtype=spec.type))
            initializer[:, 0] = initial
            return self.variable(
                name=(name.replace('/', '_') + '-buffer'),
                spec=TensorSpec(type=spec.type, shape=shape), initializer=initializer,
                is_trainable=False, is_saved=False
            )

        self.internals_buffer = self.internals_spec.fmap(
            function=function, cls=VariableDict, with_names=True, zip_values=self.internals_init
        )

    def initialize_api(self):
        if self.summarizer is not None:
            default_summarizer = self.summarizer.as_default()
        else:
            default_summarizer = util.NullContext()

        with default_summarizer:
            if self.summary_labels == 'all' or 'graph' in self.summary_labels:
                tf.summary.trace_on(graph=True, profiler=False)
            self.act(
                states=self.unprocessed_states_spec.empty(batched=True),
                auxiliaries=self.auxiliaries_spec.empty(batched=True),
                parallel=self.parallel_spec.empty(batched=True)
            )
            self.independent_act(
                states=self.unprocessed_states_spec.empty(batched=True),
                internals=self.internals_spec.empty(batched=True),
                auxiliaries=self.auxiliaries_spec.empty(batched=True)
            )
            self.observe(
                terminal=self.terminal_spec.empty(batched=True),
                reward=self.reward_spec.empty(batched=True),
                parallel=self.parallel_spec.empty(batched=False)
            )
            if self.summary_labels == 'all' or 'graph' in self.summary_labels:
                tf.summary.trace_export(name='graph', step=self.timesteps, profiler_outdir=None)

    def input_signature(self, *, function):
        if function == 'act':
            return SignatureDict(
                states=self.unprocessed_states_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                parallel=self.parallel_spec.signature(batched=True)
            )

        elif function == 'apply_exploration':
            return SignatureDict(
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True),
                exploration=self.actions_spec.fmap(
                    function=(lambda x: TensorSpec(type='float', shape=()))
                ).signature(batched=False)
            )

        elif function == 'core_act':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        elif function == 'core_observe':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True),
                terminal=self.terminal_spec.signature(batched=True),
                reward=self.reward_spec.signature(batched=True)
            )

        elif function == 'independent_act':
            return SignatureDict(
                states=self.unprocessed_states_spec.signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        elif function == 'observe':
            return SignatureDict(
                terminal=self.terminal_spec.signature(batched=True),
                reward=self.reward_spec.signature(batched=True),
                parallel=self.parallel_spec.signature(batched=False)
            )

        elif function == 'reset':
            return SignatureDict()

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=0)
    def reset(self):
        zeros = tf_util.zeros(shape=(self.parallel_interactions,), dtype='int')
        assignment = self.buffer_index.assign(value=zeros, read_value=False)

        # TODO: Synchronization optimizer initial sync?

        with tf.control_dependencies(control_inputs=(assignment,)):
            timestep = tf_util.identity(input=self.timesteps)
            episode = tf_util.identity(input=self.episodes)
            update = tf_util.identity(input=self.updates)

        return timestep, episode, update

    @tf_function(num_args=3)
    def independent_act(self, *, states, internals, auxiliaries):
        true = tf_util.constant(value=True, dtype='bool')
        batch_size = tf_util.cast(x=tf.shape(input=states.value())[0], dtype='int')

        # Input assertions
        dependencies = self.unprocessed_states_spec.tf_assert(
            x=states, batch_size=batch_size,
            message='Agent.independent_act: invalid {issue} for {name} state input.'
        )
        dependencies.extend(self.internals_spec.tf_assert(
            x=internals, batch_size=batch_size,
            message='Agent.independent_act: invalid {issue} for {name} internal input.'
        ))
        dependencies.extend(self.auxiliaries_spec.tf_assert(
            x=auxiliaries, batch_size=batch_size,
            message='Agent.independent_act: invalid {issue} for {name} input.'
        ))
        # Mask assertions
        if self.config.enable_int_action_masking:
            for name, spec in self.actions_spec.items():
                if spec.type == 'int':
                    dependencies.append(tf.debugging.assert_equal(
                        x=tf.reduce_all(input_tensor=tf.math.reduce_any(
                            input_tensor=auxiliaries[name]['mask'], axis=(spec.rank + 1)
                        )), y=true,
                        message="Agent.independent_act: at least one action has to be valid."
                    ))

        # Variable noise
        if self.config.apply_final_variable_noise and len(self.trainable_variables) > 0 and \
                self.variable_noise.final_value() != 0.0:
            with tf.control_dependencies(control_inputs=dependencies):
                dependencies = list()
                variable_noise_tensors = list()
                variable_noise = tf_util.constant(
                    value=self.variable_noise.final_value(), dtype=self.variable_noise.spec.type
                )
                for variable in self.trainable_variables:
                    noise = tf.random.normal(
                        shape=tf_util.shape(x=variable), mean=0.0, stddev=variable_noise,
                        dtype=self.variable_noise.spec.tf_type()
                    )
                    if variable.dtype != tf_util.get_dtype(type='float'):
                        noise = tf.cast(x=noise, dtype=variable.dtype)
                    dependencies.append(variable.assign_add(delta=noise, read_value=False))
                    variable_noise_tensors.append(noise)
        else:
            variable_noise_tensors = None

        with tf.control_dependencies(control_inputs=dependencies):
            # Preprocessing
            for name in self.states_spec:
                if name in self.preprocessing:
                    states[name] = self.preprocessing[name].apply(x=states[name])

            # Core act
            actions, internals = self.core_act(
                states=states, internals=internals, auxiliaries=auxiliaries, deterministic=True
            )
            # Skip action assertions

        # Variable noise
        if variable_noise_tensors is not None:
            with tf.control_dependencies(control_inputs=(actions.flatten() + internals.flatten())):
                dependencies = [
                    variable.assign_sub(delta=noise, read_value=False)
                    for variable, noise in zip(self.trainable_variables, variable_noise_tensors)
                ]
        else:
            dependencies = list()

        # Exploration
        if self.config.apply_final_exploration:
            exploration = TensorDict()
            for name in self.actions_spec:
                if isinstance(self.exploration, dict):
                    if name in self.exploration and self.exploration[name].final_value() != 0.0:
                        exploration[name] = tf_util.constant(
                            value=self.exploration[name].final_value(),
                            dtype=self.exploration[name].spec.type
                        )
                elif self.exploration.final_value() != 0.0:
                    exploration[name] = tf_util.constant(
                        value=self.exploration.final_value(), dtype=self.exploration.spec.type
                    )
            if len(exploration) > 0:
                actions = self.apply_exploration(
                    auxiliaries=auxiliaries, actions=actions, exploration=exploration
                )

        # Return
        with tf.control_dependencies(control_inputs=dependencies):
            actions.fmap(function=tf_util.identity)
            internals.fmap(function=tf_util.identity)
        return actions, internals

    @tf_function(num_args=3)
    def act(self, *, states, auxiliaries, parallel):
        true = tf_util.constant(value=True, dtype='bool')
        one = tf_util.constant(value=1, dtype='int')
        batch_size = tf_util.cast(x=tf.shape(input=parallel)[0], dtype='int')

        # Input assertions
        dependencies = self.unprocessed_states_spec.tf_assert(
            x=states, batch_size=batch_size,
            message='Agent.act: invalid {issue} for {name} state input.'
        )
        dependencies.extend(self.auxiliaries_spec.tf_assert(
            x=auxiliaries, batch_size=batch_size,
            message='Agent.act: invalid {issue} for {name} input.'
        ))
        dependencies.extend(self.parallel_spec.tf_assert(
            x=parallel, batch_size=batch_size,
            message='Agent.act: invalid {issue} for parallel input.'
        ))
        # Mask assertions
        if self.config.enable_int_action_masking:
            for name, spec in self.actions_spec.items():
                if spec.type == 'int':
                    dependencies.append(tf.debugging.assert_equal(
                        x=tf.reduce_all(input_tensor=tf.math.reduce_any(
                            input_tensor=auxiliaries[name]['mask'], axis=(spec.rank + 1)
                        )), y=true,
                        message="Agent.independent_act: at least one action has to be valid."
                    ))

        # Variable noise
        if len(self.trainable_variables) > 0 and not self.variable_noise.is_constant(value=0.0):
            with tf.control_dependencies(control_inputs=dependencies):
                zero = tf_util.constant(value=0.0, dtype='float')
                variable_noise = self.variable_noise.value()

                def no_variable_noise():
                    return [tf.zeros_like(input=variable) for variable in self.trainable_variables]

                def apply_variable_noise():
                    variable_noise_tensors = list()
                    for variable in self.trainable_variables:
                        noise = tf.random.normal(
                            shape=tf_util.shape(x=variable), mean=0.0, stddev=variable_noise,
                            dtype=self.variable_noise.spec.tf_type()
                        )
                        if variable.dtype != tf_util.get_dtype(type='float'):
                            noise = tf.cast(x=noise, dtype=variable.dtype)
                        assignment = variable.assign_add(delta=noise, read_value=False)
                        with tf.control_dependencies(control_inputs=assignment):
                            variable_noise_tensors.append(tf_util.identity(input=noise))
                    return variable_noise_tensors

                variable_noise_tensors = tf.cond(
                    pred=tf.math.equal(x=variable_noise, y=zero),
                    true_fn=no_variable_noise, false_fn=apply_variable_noise
                )
                dependencies = variable_noise_tensors

        else:
            variable_noise_tensors = None

        with tf.control_dependencies(control_inputs=dependencies):
            # Preprocessing
            for name in self.states_spec:
                if name in self.preprocessing:
                    states[name] = self.preprocessing[name].apply(x=states[name])

            # Internals
            buffer_index = tf.gather(params=self.buffer_index, indices=parallel)
            indices = tf.stack(values=(parallel, buffer_index), axis=1)
            internals = self.internals_buffer.fmap(
                function=(lambda x: tf.gather_nd(params=x, indices=indices)), cls=TensorDict
            )

            # Core act
            actions, internals = self.core_act(
                states=states, internals=internals, auxiliaries=auxiliaries, deterministic=False
            )

        # Variable noise
        if variable_noise_tensors is not None:
            with tf.control_dependencies(control_inputs=(actions.flatten() + internals.flatten())):

                def apply_variable_noise():
                    assignments = list()
                    for variable, noise in zip(self.trainable_variables, variable_noise_tensors):
                        assignments.append(variable.assign_sub(delta=noise, read_value=False))
                    return tf.group(*assignments)

                reverse_variable_noise = tf.cond(
                    pred=tf.math.equal(x=variable_noise, y=zero),
                    true_fn=tf.no_op, false_fn=apply_variable_noise
                )
                dependencies = [reverse_variable_noise]

        else:
            dependencies = list()

        # Action assertions
        dependencies.extend(self.actions_spec.tf_assert(x=actions, batch_size=batch_size))
        if self.config.enable_int_action_masking:
            for name, spec, action in self.actions_spec.zip_items(actions):
                if spec.type == 'int':
                    is_valid = tf.reduce_all(input_tensor=tf.gather(
                        params=auxiliaries[name]['mask'],
                        indices=tf.expand_dims(input=action, axis=(spec.rank + 1)),
                        batch_dims=(spec.rank + 1)
                    ))
                    dependencies.append(tf.debugging.assert_equal(
                        x=is_valid, y=true, message="Action mask check."
                    ))

        # Exploration
        exploration = TensorDict()
        for name in self.actions_spec:
            if isinstance(self.exploration, dict):
                if name not in self.exploration:
                    continue
                if not self.exploration[name].is_constant(value=0.0):
                    exploration[name] = self.exploration[name].value()
            elif not self.exploration.is_constant(value=0.0):
                exploration[name] = self.exploration.value()
        if len(exploration) > 0:
            actions = self.apply_exploration(
                auxiliaries=auxiliaries, actions=actions, exploration=exploration
            )

        # Action assertions
        dependencies.extend(self.actions_spec.tf_assert(x=actions, batch_size=batch_size))
        if self.config.enable_int_action_masking:
            for name, spec in self.actions_spec.items():
                if spec.type == 'int':
                    is_valid = tf.reduce_all(input_tensor=tf.gather(
                        params=auxiliaries[name]['mask'],
                        indices=tf.expand_dims(input=actions[name], axis=(spec.rank + 1)),
                        batch_dims=(spec.rank + 1)
                    ))
                    dependencies.append(tf.debugging.assert_equal(
                        x=is_valid, y=true, message="Action mask check."
                    ))

        # Update states/internals/actions buffers
        buffer_index = tf.gather(params=self.buffer_index, indices=parallel)
        indices = tf.stack(values=(parallel, buffer_index), axis=1)
        for name, buffer, state in self.states_buffer.zip_items(states):
            dependencies.append(buffer.scatter_nd_update(indices=indices, updates=state))
        for name, buffer, auxiliary in self.auxiliaries_buffer.zip_items(auxiliaries):
            dependencies.append(buffer.scatter_nd_update(indices=indices, updates=auxiliary))
        for name, buffer, action in self.actions_buffer.zip_items(actions):
            dependencies.append(buffer.scatter_nd_update(indices=indices, updates=action))
        indices = tf.stack(values=(parallel, buffer_index + one), axis=1)
        for name, buffer, internal in self.internals_buffer.zip_items(internals):
            dependencies.append(buffer.scatter_nd_update(indices=indices, updates=internal))

        # Increment timesteps and buffer index 
        with tf.control_dependencies(control_inputs=dependencies):
            dependencies = list()
            dependencies.append(self.timesteps.assign_add(delta=batch_size, read_value=False))
            # dependencies.append(self.parallel_timestep.scatter_nd_add(
            #     indices=tf.expand_dims(input=parallel, axis=1), updates=tf.ones_like(input=parallel)
            # ))
            dependencies.append(self.buffer_index.scatter_nd_add(
                indices=tf.expand_dims(input=parallel, axis=1), updates=tf.ones_like(input=parallel)
            ))

        # Return
        with tf.control_dependencies(control_inputs=dependencies):
            actions = actions.fmap(function=tf_util.identity)
            timestep = tf_util.identity(input=self.timesteps)
        return actions, timestep

    @tf_function(num_args=3)
    def observe(self, *, terminal, reward, parallel):
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        batch_size = tf_util.cast(x=tf.shape(input=terminal)[0], dtype='int')
        expanded_parallel = tf.expand_dims(input=tf.expand_dims(input=parallel, axis=0), axis=1)
        buffer_index = tf.gather(params=self.buffer_index, indices=parallel)
        is_terminal = tf.concat(values=([zero], terminal), axis=0)[-1] > zero

        # Input assertions
        dependencies = self.terminal_spec.tf_assert(
            x=terminal, batch_size=batch_size,
            message='Agent.observe: invalid {issue} for terminal input.'
        )
        dependencies.extend(self.reward_spec.tf_assert(
            x=reward, batch_size=batch_size,
            message='Agent.observe: invalid {issue} for terminal input.'
        ))
        dependencies.extend(self.parallel_spec.tf_assert(
            x=parallel, message='Agent.observe: invalid {issue} for parallel input.'
        ))
        # Assertion: size of terminal equals buffer index
        dependencies.append(tf.debugging.assert_equal(
            x=tf_util.cast(x=tf.shape(input=terminal)[0], dtype='int'), y=buffer_index,
            message="Agent.observe: number of observe-timesteps has to be equal to number of "
                    "buffered act-timesteps."
        ))
        # Assertion: at most one terminal
        dependencies.append(tf.debugging.assert_less_equal(
            x=tf_util.cast(x=tf.math.count_nonzero(input=terminal), dtype='int'), y=one,
            message="Agent.observe: input contains more than one terminal."
        ))
        # Assertion: if terminal, last timestep in batch
        dependencies.append(tf.debugging.assert_equal(
            x=tf.math.reduce_any(input_tensor=tf.math.greater(x=terminal, y=zero)), y=is_terminal,
            message="Agent.observe: terminal is not the last input timestep."
        ))

        with tf.control_dependencies(control_inputs=dependencies):
            # Reward summary
            if self.summary_labels == 'all' or 'reward' in self.summary_labels:
                with self.summarizer.as_default():
                    x = tf.math.reduce_mean(input_tensor=reward)
                    tf.summary.scalar(name='reward', data=x, step=self.timesteps)

            # Update episode reward
            episode_reward = tf.math.reduce_sum(input_tensor=reward, keepdims=True)
            assignment = self.episode_reward.scatter_nd_add(
                indices=expanded_parallel, updates=episode_reward
            )
            dependencies = [assignment]

        # Reward preprocessing (after episode_reward update)
        if 'reward' in self.preprocessing:
            with tf.control_dependencies(control_inputs=dependencies):
                reward = self.preprocessing['reward'].apply(x=reward)

                # Preprocessed reward summary
                if self.summary_labels == 'all' or 'reward' in self.summary_labels:
                    with self.summarizer.as_default():
                        x = tf.math.reduce_mean(input_tensor=reward)
                        tf.summary.scalar(name='preprocessed-reward', data=x, step=self.timesteps)

                # Update preprocessed episode reward
                episode_reward = tf.math.reduce_sum(input_tensor=reward, keepdims=True)
                assignment = self.preprocessed_episode_reward.scatter_nd_add(
                    indices=expanded_parallel, updates=episode_reward
                )
                dependencies = [assignment]

        # Handle terminal (after preprocessed_episode_reward update)
        with tf.control_dependencies(control_inputs=dependencies):

            def fn_is_terminal():
                operations = list()

                # Episode reward summaries (before episode reward reset / episodes increment)
                if self.summary_labels == 'all' or 'reward' in self.summary_labels:
                    with self.summarizer.as_default():
                        x = tf.gather(params=self.episode_reward, indices=parallel)
                        tf.summary.scalar(name='episode-reward', data=x, step=self.episodes)
                        if 'reward' in self.preprocessing:
                            x = tf.gather(params=self.preprocessed_episode_reward, indices=parallel)
                            tf.summary.scalar(
                                name='preprocessed-episode-reward', data=x, step=self.episodes
                            )

                # Reset episode reward
                zero_float = tf_util.constant(value=0.0, dtype='float', shape=(1,))
                operations.append(self.episode_reward.scatter_nd_update(
                    indices=expanded_parallel, updates=zero_float
                ))
                if 'reward' in self.preprocessing:
                    operations.append(self.preprocessed_episode_reward.scatter_nd_update(
                        indices=expanded_parallel, updates=zero_float
                    ))

                # Increment episodes counter
                operations.append(self.episodes.assign_add(delta=one, read_value=False))

                # Reset preprocessors
                for preprocessor in self.preprocessing.values():
                    operations.append(preprocessor.reset())

                return tf.group(*operations)

            handle_terminal = tf.cond(pred=is_terminal, true_fn=fn_is_terminal, false_fn=tf.no_op)

        with tf.control_dependencies(control_inputs=(handle_terminal,)):
            # Values from buffers
            function = (lambda x: x[parallel, :buffer_index])
            states = self.states_buffer.fmap(function=function, cls=TensorDict)
            internals = self.internals_buffer.fmap(function=function, cls=TensorDict)
            auxiliaries = self.auxiliaries_buffer.fmap(function=function, cls=TensorDict)
            actions = self.actions_buffer.fmap(function=function, cls=TensorDict)

            # Core observe
            updated = self.core_observe(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                terminal=terminal, reward=reward
            )

        # Reset buffer index (independent of rest)
        with tf.control_dependencies(control_inputs=(buffer_index,)):
            reset_buffer_index = self.buffer_index.scatter_nd_update(
                indices=expanded_parallel, updates=tf.expand_dims(input=zero, axis=0)
            )

        # Return
        with tf.control_dependencies(control_inputs=(updated, reset_buffer_index)):
            updated = tf_util.identity(input=updated)
            episodes = tf_util.identity(input=self.episodes)
            updates = tf_util.identity(input=self.updates)

        return updated, episodes, updates

    @tf_function(num_args=3)
    def core_act(self, *, states, internals, auxiliaries, deterministic):
        raise NotImplementedError

    @tf_function(num_args=6)
    def core_observe(self, *, states, internals, auxiliaries, actions, terminal, reward):
        raise NotImplementedError

    @tf_function(num_args=3)
    def apply_exploration(self, *, auxiliaries, actions, exploration):
        float_dtype = tf_util.get_dtype(type='float')
        for name, spec, action, exploration in self.actions_spec.zip_items(actions, exploration):
            zero = tf_util.constant(value=0.0, dtype='float')

            def no_exploration():
                return action

            def apply_exploration():
                shape = tf_util.cast(x=tf.shape(input=action), dtype='int')

                if spec.type == 'bool':
                    # Bool action: if uniform[0, 1] < exploration, then uniform[True, False]
                    half = tf_util.constant(value=0.5, dtype='float')
                    random_action = tf.random.uniform(shape=shape, dtype=float_dtype) < half
                    is_random = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                    return tf.where(condition=is_random, x=random_action, y=action)

                elif spec.type == 'int' and spec.num_values is not None:
                    if self.config.enable_int_action_masking:
                        # Masked action: if uniform[0, 1] < exploration, then uniform[unmasked]
                        # (Similar code as for RandomModel.core_act)
                        mask = auxiliaries[name]['mask']
                        choices = tf_util.constant(
                            value=list(range(spec.num_values)), dtype=spec.type,
                            shape=(tuple(1 for _ in spec.shape) + (1, spec.num_values))
                        )
                        one = tf_util.constant(value=1, dtype='int', shape=(1,))
                        multiples = tf.concat(values=(shape, one), axis=0)
                        choices = tf.tile(input=choices, multiples=multiples)
                        choices = tf.boolean_mask(tensor=choices, mask=mask)
                        mask = tf_util.cast(x=mask, dtype='int')
                        num_unmasked = tf.math.reduce_sum(input_tensor=mask, axis=(spec.rank + 1))
                        masked_offset = tf.math.cumsum(x=num_unmasked, axis=spec.rank, exclusive=True)
                        uniform = tf.random.uniform(shape=shape, dtype=float_dtype)
                        num_unmasked = tf_util.cast(x=num_unmasked, dtype='float')
                        random_offset = tf_util.cast(x=(uniform * num_unmasked), dtype='int')
                        random_action = tf.gather(
                            params=choices, indices=(masked_offset + random_offset)
                        )
                        is_random = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                        return tf.where(condition=is_random, x=random_action, y=action)

                    else:
                        # Bounded int action: if uniform[0, 1] < exploration, then uniform[num_values]
                        random_action = tf.random.uniform(
                            shape=shape, maxval=spec.num_values, dtype=spec.tf_type()
                        )
                        is_random = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                        return tf.where(condition=is_random, x=random_action, y=action)

                else:
                    # Int/float action: action + normal[0, exploration]
                    noise = tf.random.normal(shape=shape, dtype=spec.tf_type())
                    _action = action + noise * exploration

                    # Clip action if left-/right-bounded
                    if spec.min_value is not None:
                        _action = tf.math.maximum(x=_action, y=spec.min_value)
                    if spec.max_value is not None:
                        _action = tf.math.minimum(x=_action, y=spec.max_value)
                    return _action

            actions[name] = tf.cond(
                pred=tf.math.equal(x=exploration, y=zero),
                true_fn=no_exploration, false_fn=apply_exploration
            )

        return actions

    def get_variable(self, *, variable):
        assert False, 'Not updated yet!'
        if not variable.startswith(self.name):
            variable = util.join_scopes(self.name, variable)
        fetches = variable + '-output:0'
        return self.monitored_session.run(fetches=fetches)

    def assign_variable(self, *, variable, value):
        if variable.startswith(self.name + '/'):
            variable = variable[len(self.name) + 1:]
        module = self
        scope = variable.split('/')
        for _ in range(len(scope) - 1):
            module = module.modules[scope.pop(0)]
        fetches = util.join_scopes(self.name, variable) + '-assign'
        dtype = util.dtype(x=module.variables[scope[0]])
        feed_dict = {util.join_scopes(self.name, 'assignment-') + dtype + '-input:0': value}
        self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)

    def summarize(self, *, summary, value, step=None):
        fetches = util.join_scopes(self.name, summary, 'write_summary', 'Const:0')
        feed_dict = {util.join_scopes(self.name, 'summarize-input:0'): value}
        if step is not None:
            feed_dict[util.join_scopes(self.name, 'summarize-step-input:0')] = step
        self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)


        # if self.summarizer_spec is not None:
        #     if len(self.summarizer_spec.get('custom', ())) > 0:
        #         self.summarize_input = self.add_placeholder(
        #             name='summarize', dtype='float', shape=None, batched=False
        #         )
        #         # self.summarize_step_input = self.add_placeholder(
        #         #     name='summarize-step', dtype='int', shape=(), batched=False,
        #         #     default=self.timesteps
        #         # )
        #         self.summarize_step_input = self.timesteps
        #         self.custom_summaries = OrderedDict()
        #         for name, summary in self.summarizer_spec['custom'].items():
        #             if summary['type'] == 'audio':
        #                 self.custom_summaries[name] = tf.summary.audio(
        #                     name=name, data=self.summarize_input,
        #                     sample_rate=summary['sample_rate'],
        #                     step=self.summarize_step_input,
        #                     max_outputs=summary.get('max_outputs', 3),
        #                     encoding=summary.get('encoding')
        #                 )
        #             elif summary['type'] == 'histogram':
        #                 self.custom_summaries[name] = tf.summary.histogram(
        #                     name=name, data=self.summarize_input,
        #                     step=self.summarize_step_input,
        #                     buckets=summary.get('buckets')
        #                 )
        #             elif summary['type'] == 'image':
        #                 self.custom_summaries[name] = tf.summary.image(
        #                     name=name, data=self.summarize_input,
        #                     step=self.summarize_step_input,
        #                     max_outputs=summary.get('max_outputs', 3)
        #                 )
        #             elif summary['type'] == 'scalar':
        #                 self.custom_summaries[name] = tf.summary.scalar(
        #                     name=name,
        #                     data=tf.reshape(tensor=self.summarize_input, shape=()),
        #                     step=self.summarize_step_input
        #                 )
        #             else:
        #                 raise TensorforceError.value(
        #                     name='custom summary', argument='type', value=summary['type'],
        #                     hint='not in {audio,histogram,image,scalar}'
        #                 )


    def save(self, *, directory=None, filename=None, format='checkpoint', append=None):
        if directory is None and filename is None and format == 'checkpoint':
            if self.saver is None:
                raise TensorforceError.required(name='Model.save', argument='directory')
            if append is None:
                append = self.saver._step_counter
            else:
                append = self.units[append]
            return self.saver.save(checkpoint_number=append)

        if directory is None:
            raise TensorforceError.required(name='Model.save', argument='directory')

        if append is not None:
            append_value = self.units[append].numpy().item()

        if format == 'saved-model':
            if filename is not None:
                raise TensorforceError.invalid(name='Model.save', argument='filename')
            if append is not None:
                directory = os.path.join(directory, append[:-1] + str(append_value))
            assert hasattr(self, '_independent_act_graphs')
            assert len(self._independent_act_graphs) == 1
            independent_act = next(iter(self._independent_act_graphs.values()))
            return tf.saved_model.save(obj=self, export_dir=directory, signatures=independent_act)

        if filename is None:
            filename = self.name

        if append is not None:
            filename = filename + '-' + str(append_value)

        if format == 'checkpoint':
            # which variables are not saved? should all be saved probably, so remove option
            # always write temporary terminal=2/3 to indicate it is in process... has been removed recently...
            # check everywhere temrinal is checked that this is correct, if 3 is used.
            # Reset should reset estimator!!!
            return super().save(directory=directory, filename=filename)

        # elif format == 'tensorflow':
        #     if self.summarizer_spec is not None:
        #         self.monitored_session.run(fetches=self.summarizer_flush)
        #     saver_path = self.saver.save(
        #         sess=self.session, save_path=path, global_step=append,
        #         # latest_filename=None,  # Defaults to 'checkpoint'.
        #         meta_graph_suffix='meta', write_meta_graph=True, write_state=True
        #     )
        #     assert saver_path.startswith(path)
        #     path = saver_path

        #     if not no_act_pb:
        #         graph_def = self.graph.as_graph_def()

        #         # freeze_graph clear_devices option
        #         for node in graph_def.node:
        #             node.device = ''

        #         graph_def = tf.compat.v1.graph_util.remove_training_nodes(input_graph=graph_def)
        #         output_node_names = [
        #             self.name + '.independent_act/' + name + '-output'
        #             for name in self.output_tensors['independent_act']
        #         ]
        #         # implies tf.compat.v1.graph_util.extract_sub_graph
        #         graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        #             sess=self.monitored_session, input_graph_def=graph_def,
        #             output_node_names=output_node_names
        #         )
        #         graph_path = tf.io.write_graph(
        #             graph_or_graph_def=graph_def, logdir=directory,
        #             name=(os.path.split(path)[1] + '.pb'), as_text=False
        #         )
        #         assert graph_path == path + '.pb'
        #     return path

        elif format == 'numpy':
            variables = dict()
            for variable in self.saved_variables:
                variables[variable.name[len(self.name) + 1: -2]] = variable.numpy()
            path = os.path.join(directory, filename) + '.npz'
            np.savez(file=path, **variables)
            return path

        elif format == 'hdf5':
            path = os.path.join(directory, filename) + '.hdf5'
            with h5py.File(name=path, mode='w') as filehandle:
                for variable in self.saved_variables:
                    name = variable.name[len(self.name) + 1: -2]
                    filehandle.create_dataset(name=name, data=variable.numpy())
            return path

        else:
            raise TensorforceError.value(name='Model.save', argument='format', value=format)

    def restore(self, *, directory=None, filename=None, format='checkpoint'):
        if format == 'checkpoint':
            if directory is None:
                if self.saver is None:
                    raise TensorforceError.required(name='Model.save', argument='directory')
                directory = self.saver_directory
            if filename is None:
                filename = tf.train.latest_checkpoint(checkpoint_dir=directory)
                _directory, filename = os.path.split(filename)
                assert _directory == directory
            super().restore(directory=directory, filename=filename)

        elif format == 'saved-model':
            # TODO: Check memory/estimator/etc variables are not included!
            raise TensorforceError.value(name='Model.load', argument='format', value=format)

        # elif format == 'tensorflow':
        #     self.saver.restore(sess=self.session, save_path=path)

        elif format == 'numpy':
            if directory is None:
                raise TensorforceError(
                    name='Model.load', argument='directory', condition='format is "numpy"'
                )
            if filename is None:
                raise TensorforceError(
                    name='Model.load', argument='filename', condition='format is "numpy"'
                )
            variables = np.load(file=(os.path.join(directory, filename) + '.npz'))
            for variable in self.saved_variables:
                variable.assign(value=variables[variable.name[len(self.name) + 1: -2]])

        elif format == 'hdf5':
            if directory is None:
                raise TensorforceError(
                    name='Model.load', argument='directory', condition='format is "hdf5"'
                )
            if filename is None:
                raise TensorforceError(
                    name='Model.load', argument='filename', condition='format is "hdf5"'
                )
            path = os.path.join(directory, filename)
            if os.path.isfile(path + '.hdf5'):
                path = path + '.hdf5'
            else:
                path = path + '.h5'
            with h5py.File(name=path, mode='r') as filehandle:
                for variable in self.saved_variables:
                    variable.assign(value=filehandle[variable.name[len(self.name) + 1: -2]])

        else:
            raise TensorforceError.value(name='Model.load', argument='format', value=format)

        return self.timesteps.numpy().item(), self.episodes.numpy().item(), \
            self.updates.numpy().item()
