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

import os

import h5py
import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import ArrayDict, Module, ModuleDict, parameter_modules, TensorDict, \
    TensorSpec, TensorsSpec, tf_function, tf_util, VariableDict
from tensorforce.core.networks import Preprocessor


class Model(Module):

    def __init__(
        self,
        # Model
        states, actions, preprocessing, exploration, variable_noise, l2_regularization, name,
        device, parallel_interactions, saver, summarizer, config
    ):
        # TODO: summarizer/saver/etc should be part of module init?
        if summarizer is None or summarizer.get('directory') is None:
            summary_labels = None
        else:
            summary_labels = summarizer.get('labels', ('graph',))

        super().__init__(
            name=name, is_root=True, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        # Tensorforce config
        # TODO: should be part of Module init?
        self.config = config

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
        if self.config.enable_int_action_masking:
            for name, spec in self.actions_spec.items():
                if spec.type == 'int':
                    self.auxiliaries_spec[name + '_mask'] = TensorSpec(
                        type='bool', shape=(spec.shape + (spec.num_values,))
                    )

            # Check for name collisions
            for name in self.auxiliaries_spec:
                if name in self.value_names:
                    raise TensorforceError.exists(name='value name', value=name)
                self.value_names.add(name)

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
                self.preprocessing[name] = self.add_module(
                    name=(name + '_preprocessing'), module=Preprocessor, is_trainable=False,
                    input_spec=spec, layers=layers
                )
                self.states_spec[name] = self.preprocessing[name].get_output_spec()

        # Reward preprocessing
        if preprocessing is not None:
            if 'reward' in preprocessing:
                self.preprocessing['reward'] = self.add_module(
                    name=('reward_preprocessing'), module=Preprocessor, is_trainable=False,
                    input_spec=self.reward_spec, layers=preprocessing['reward']
                )
                if self.preprocessing['reward'].get_output_spec() != self.reward_spec:
                    raise TensorforceError.mismatch(
                        name='preprocessing', argument='reward output spec',
                        value1=self.preprocessing['reward'].get_output_spec(),
                        value2=self.reward_spec
                    )

        # Exploration
        # TODO: recursive lookup???
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
                    self.exploration[name] = self.add_module(
                        name=(name + '_exploration'), module=module, modules=parameter_modules,
                        is_trainable=False, dtype='float', min_value=0.0, max_value=1.0
                    )
                else:
                    self.exploration[name] = self.add_module(
                        name=(name + '_exploration'), module=module, modules=parameter_modules,
                        is_trainable=False, dtype='float', min_value=0.0
                    )
        else:
            # Same exploration for all actions
            self.exploration = self.add_module(
                name='exploration', module=exploration, modules=parameter_modules,
                is_trainable=False, dtype='float', min_value=0.0
            )

        # Variable noise
        self.variable_noise = self.add_module(
            name='variable_noise', module=variable_noise, modules=parameter_modules,
            is_trainable=False, dtype='float', min_value=0.0
        )

        # Parallel interactions
        assert isinstance(parallel_interactions, int) and parallel_interactions >= 1
        self.parallel_interactions = parallel_interactions

        # Saver
        if saver is None:
            self.saver_spec = None
        elif not all(
            key in ('directory', 'filename', 'frequency', 'load', 'max-checkpoints')
            for key in saver
        ):
            raise TensorforceError.value(
                name='agent', argument='saver', value=list(saver),
                hint='not from {directory,filename,frequency,load,max-checkpoints}'
            )
        elif saver.get('directory') is None:
            self.saver_spec = None
        else:
            self.saver_spec = dict(saver)

        # Summarizer
        if summarizer is None:
            self.summarizer_spec = None
        elif not all(
            key in ('custom', 'directory', 'flush', 'frequency', 'labels', 'max-summaries')
            for key in summarizer
        ):
            raise TensorforceError.value(
                name='agent', argument='summarizer', value=list(summarizer),
                hint='not from {custom,directory,flush,frequency,labels,max-summaries}'
            )
        elif summarizer.get('directory') is None:
            self.summarizer_spec = None
        else:
            self.summarizer_spec = dict(summarizer)

    def input_signature(self, function):
        if function == 'act':
            return [
                self.unprocessed_states_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True),
                self.parallel_spec.signature(batched=True)
            ]

        elif function == 'apply_exploration':
            return [
                self.auxiliaries_spec.signature(batched=True),
                self.actions_spec.signature(batched=True),
                self.actions_spec.fmap(function=(lambda x: TensorSpec(type='float', shape=())))
                    .signature(batched=False)
            ]

        elif function == 'core_act':
            return [
                self.states_spec.signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True)
            ]

        elif function == 'core_observe':
            return [
                self.states_spec.signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True),
                self.actions_spec.signature(batched=True),
                self.terminal_spec.signature(batched=True),
                self.reward_spec.signature(batched=True)
            ]

        elif function == 'independent_act':
            return [
                self.states_spec.signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True)
            ]

        elif function == 'observe':
            return [
                self.terminal_spec.signature(batched=True),
                self.reward_spec.signature(batched=True),
                self.parallel_spec.signature(batched=False)
            ]

        elif function == 'reset':
            return ()

        else:
            return super().input_signature(function=function)

    def close(self):
        if self.summarizer_spec is not None:
            self.monitored_session.run(fetches=self.summarizer_close)
        if self.saver_directory is not None:
            self.save(
                directory=self.saver_directory, filename=self.saver_filename, format='tensorflow',
                append='timesteps', no_act_pb=True
            )

    def initialize(self):
        super().initialize()

        # Episode reward
        self.episode_reward = self.variable(
            name='episode-reward', dtype=self.reward_spec.type, shape=(self.parallel_interactions,),
            initializer='zeros', is_trainable=False, is_saved=False
        )

        # Buffer index
        self.buffer_index = self.variable(
            name='buffer-index', dtype='int', shape=(self.parallel_interactions,),
            initializer='zeros', is_trainable=False, is_saved=False
        )

        # States/auxiliaries/actions buffers
        function = (lambda name, spec: self.variable(
            name=(name.replace('/', '_') + '-buffer'), dtype=spec.type,
            shape=((self.parallel_interactions, self.config.buffer_observe) + spec.shape),
            initializer='zeros', is_trainable=False, is_saved=False
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
        self.internals_buffer = VariableDict()
        for name, spec, initial in util.zip_items(self.internals_spec, self.internals_init):
            shape = ((self.parallel_interactions, self.config.buffer_observe + 1) + spec.shape)
            initializer = np.zeros(shape=shape, dtype=util.np_dtype(dtype=spec.type))
            initializer[:, 0] = initial
            self.internals_buffer[name] = self.variable(
                name=(name.replace('/', '_') + '-buffer'), dtype=spec.type, shape=shape,
                initializer=initializer, is_trainable=False, is_saved=False
            )

    @tf_function(num_args=0)
    def reset(self):
        zeros = tf_util.zeros(shape=(self.parallel_interactions,), dtype='int')
        assignment = self.buffer_index.assign(value=zeros, read_value=False)

        # TODO: Synchronization initial sync?

        with tf.control_dependencies(control_inputs=(assignment,)):
            timestep = tf_util.identity(input=self.timesteps)
            episode = tf_util.identity(input=self.episodes)
            update = tf_util.identity(input=self.updates)

        return timestep, episode, update

    @tf_function(num_args=3)
    def independent_act(self, states, internals, auxiliaries):
        true = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
        batch_size = tf.shape(input=states.item())[0]

        # Input assertions
        dependencies = self.states_spec.tf_assert(
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
                            input_tensor=auxiliaries[name + '_mask'], axis=(spec.rank + 1)
                        )), y=true,
                        message="Agent.independent_act: at least one action has to be valid."
                    ))

        # Variable noise
        if len(self.trainable_variables) > 0 and self.variable_noise.final_value() > 0.0:
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
            dependencies = util.flatten(xs=actions) + util.flatten(xs=internals)
            # Skip action assertions

        # Variable noise
        if variable_noise_tensors is not None:
            with tf.control_dependencies(control_inputs=dependencies):
                dependencies = [
                    variable.assign_sub(delta=noise, read_value=False)
                    for variable, noise in zip(self.trainable_variables, variable_noise_tensors)
                ]

        # Exploration
        exploration = TensorsDict()
        for name in self.actions_spec:
            if isinstance(self.exploration, dict):
                if name in self.exploration and self.exploration[name].final_value() > 0.0:
                    exploration[name] = tf_util.constant(
                        value=self.exploration[name].final_value(),
                        dtype=self.exploration[name].spec.type
                    )
            elif self.exploration.final_value() > 0.0:
                exploration[name] = tf_util.constant(
                    value=self.exploration.final_value(), dtype=self.exploration.spec.type
                )
        if len(exploration) > 0:
            with tf.control_dependencies(control_inputs=dependencies):
                actions = self.apply_exploration(
                    auxiliaries=auxiliaries, actions=actions, exploration=exploration
                )
                dependencies = util.flatten(xs=actions)

        # Return
        with tf.control_dependencies(control_inputs=dependencies):
            actions.fmap(function=tf_util.identity)
            internals.fmap(function=tf_util.identity)
        return actions, internals

    @tf_function(num_args=3)
    def act(self, states, auxiliaries, parallel):
        true = tf_util.constant(value=True, dtype='bool')
        one = tf_util.constant(value=1, dtype='int')
        batch_size = tf_util.cast(x=tf.shape(input=parallel)[0], dtype='int')

        # Input assertions
        dependencies = self.states_spec.tf_assert(
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
                            input_tensor=auxiliaries[name + '_mask'], axis=(spec.rank + 1)
                        )), y=true,
                        message="Agent.independent_act: at least one action has to be valid."
                    ))

        # Variable noise
        if len(self.trainable_variables) > 0 and self.variable_noise.max_value() > 0.0:
            with tf.control_dependencies(control_inputs=dependencies):
                dependencies = list()
                variable_noise_tensors = list()
                variable_noise = self.variable_noise.value()
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
            dependencies = util.flatten(xs=actions) + util.flatten(xs=internals)

            # Action assertions
            dependencies.extend(self.actions_spec.tf_assert(x=actions, batch_size=batch_size))
            if self.config.enable_int_action_masking:
                for name, spec in self.actions_spec.items():
                    if spec.type == 'int':
                        is_valid = tf.reduce_all(input_tensor=tf.gather(
                            params=auxiliaries[name + '_mask'],
                            indices=tf.expand_dims(input=actions[name], axis=(spec.rank + 1)),
                            batch_dims=(spec.rank + 1)
                        ))
                        dependencies.append(tf.debugging.assert_equal(
                            x=is_valid, y=true, message="Action mask check."
                        ))

        # Variable noise
        if variable_noise_tensors is not None:
            with tf.control_dependencies(control_inputs=dependencies):
                dependencies = [
                    variable.assign_sub(delta=noise, read_value=False)
                    for variable, noise in zip(self.trainable_variables, variable_noise_tensors)
                ]

        # Exploration
        exploration = TensorDict()
        for name in self.actions_spec:
            if isinstance(self.exploration, dict):
                if name in self.exploration and self.exploration[name].max_value() > 0.0:
                    exploration[name] = self.exploration[name].value()
            elif self.exploration.max_value() > 0.0:
                exploration[name] = self.exploration.value()
        if len(exploration) > 0:
            with tf.control_dependencies(control_inputs=dependencies):
                actions = self.apply_exploration(
                    auxiliaries=auxiliaries, actions=actions, exploration=exploration
                )
                dependencies = util.flatten(xs=actions)

        # Action assertions
        dependencies.extend(self.actions_spec.tf_assert(x=actions, batch_size=batch_size))
        if self.config.enable_int_action_masking:
            for name, spec in self.actions_spec.items():
                if spec.type == 'int':
                    is_valid = tf.reduce_all(input_tensor=tf.gather(
                        params=auxiliaries[name + '_mask'],
                        indices=tf.expand_dims(input=actions[name], axis=(spec.rank + 1)),
                        batch_dims=(spec.rank + 1)
                    ))
                    dependencies.append(tf.debugging.assert_equal(
                        x=is_valid, y=true, message="Action mask check."
                    ))

        # Update states/internals/actions buffers
        with tf.control_dependencies(control_inputs=dependencies):
            dependencies = list()
            buffer_index = tf.gather(params=self.buffer_index, indices=parallel)
            indices = tf.stack(values=(parallel, buffer_index), axis=1)
            for name, state, buffer in util.zip_items(states, self.states_buffer):
                dependencies.append(buffer.scatter_nd_update(indices=indices, updates=state))
            for name, auxiliary, buffer in util.zip_items(auxiliaries, self.auxiliaries_buffer):
                dependencies.append(buffer.scatter_nd_update(indices=indices, updates=auxiliary))
            for name, action, buffer in util.zip_items(actions, self.actions_buffer):
                dependencies.append(buffer.scatter_nd_update(indices=indices, updates=action))
            indices = tf.stack(values=(parallel, buffer_index + one), axis=1)
            for name, internal, buffer in util.zip_items(internals, self.internals_buffer):
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
    def observe(self, terminal, reward, parallel):
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        buffer_index = tf.gather(params=self.buffer_index, indices=parallel)
        batch_size = tf_util.cast(x=tf.shape(input=terminal)[0], dtype='int')

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
            x=tf.math.reduce_any(input_tensor=tf.math.greater(x=terminal, y=zero)),
            y=tf.math.greater(x=terminal[-1], y=zero),
            message="Agent.observe: terminal is not the last input timestep."
        ))
        # Assertion: single parallel value
        dependencies.append(tf.debugging.assert_equal(
            x=tf_util.cast(x=tf.shape(input=parallel)[0], dtype='int'), y=one,
            message="Agent.observe: parallel contains more than one value."
        ))

        with tf.control_dependencies(control_inputs=dependencies):
            dependencies = list()
            reward = self.add_summary(label=('reward', 'rewards'), name='reward', tensor=reward)
            dependencies.append(self.episode_reward.scatter_nd_add(
                indices=tf.expand_dims(input=parallel, axis=1),
                updates=tf.math.reduce_sum(input_tensor=reward, keepdims=True)
            ))

            # Reset episode reward
            # TODO: Check whether multiple writes overwrite the summary value
            def reset_episode_reward():
                zero_float = tf_util.constant(value=0.0, dtype='float', shape=(1,))
                zero_float = self.add_summary(
                    label=('episode-reward', 'rewards'), name='episode-reward',
                    tensor=tf.gather(params=self.episode_reward, indices=parallel),
                    pass_tensors=zero_float
                )
                assignment = self.episode_reward.scatter_nd_update(
                    indices=tf.expand_dims(input=parallel, axis=1), updates=zero_float
                )
                with tf.control_dependencies(control_inputs=(assignment,)):
                    return tf.no_op()

            dependencies.append(tf.cond(
                pred=(terminal[-1] > zero), true_fn=reset_episode_reward, false_fn=tf.no_op
            ))

        with tf.control_dependencies(control_inputs=dependencies):
            dependencies = list()

            # Increment episode
            def increment_episode():
                return self.episodes.assign_add(delta=one, read_value=False)

            dependencies.append(tf.cond(
                pred=(terminal[-1] > zero), true_fn=increment_episode, false_fn=tf.no_op
            ))

            # Reward preprocessing
            if 'reward' in self.preprocessing:
                with tf.control_dependencies(control_inputs=dependencies):
                    reward = self.preprocessing['reward'].apply(x=reward)
                    reward = self.add_summary(
                        label=('reward', 'rewards'), name='preprocessed-reward', tensor=reward
                    )
                    dependencies.append(reward)

        with tf.control_dependencies(control_inputs=dependencies):
            # Values from buffers
            states = self.states_buffer.fmap(
                function=(lambda x: x[parallel[0], :buffer_index[0]]), cls=TensorDict
            )
            internals = self.internals_buffer.fmap(
                function=(lambda x: x[parallel[0], :buffer_index[0]]), cls=TensorDict
            )
            auxiliaries = self.auxiliaries_buffer.fmap(
                function=(lambda x: x[parallel[0], :buffer_index[0]]), cls=TensorDict
            )
            actions = self.actions_buffer.fmap(
                function=(lambda x: x[parallel[0], :buffer_index[0]]), cls=TensorDict
            )

            # Core observe
            is_updated = self.core_observe(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                terminal=terminal, reward=reward
            )
            dependencies = [is_updated]

        with tf.control_dependencies(control_inputs=dependencies):
            dependencies = list()

            # Reset buffer index
            dependencies.append(self.buffer_index.scatter_nd_update(
                indices=parallel, updates=tf.expand_dims(input=zero, axis=0)
            ))

            # Reset preprocessors
            if len(self.preprocessing) > 0:
                def reset_preprocessors():
                    return tf.group(
                        *(preprocessor.reset() for preprocessor in self.preprocessing.values())
                    )

                dependencies.append(tf.cond(
                    pred=(terminal[-1] > zero), true_fn=reset_preprocessors, false_fn=tf.no_op
                ))

        # Return
        with tf.control_dependencies(control_inputs=dependencies):
            updated = tf_util.identity(input=is_updated)
            episode = tf_util.identity(input=self.episodes)
            update = tf_util.identity(input=self.updates)

        return updated, episode, update

    @tf_function(num_args=3)
    def core_act(self, states, internals, auxiliaries, deterministic):
        raise NotImplementedError

    @tf_function(num_args=6)
    def core_observe(self, states, internals, auxiliaries, actions, terminal, reward):
        raise NotImplementedError

    @tf_function(num_args=3)
    def apply_exploration(self, auxiliaries, actions, exploration):
        for name, spec in self.actions_spec.items():
            if name not in exploration:
                continue
            shape = tf_util.cast(x=tf.shape(input=actions[name]), dtype='int')
            float_dtype = tf_util.get_dtype(type='float')

            if spec.type == 'bool':
                # Bool action: if uniform[0, 1] < exploration, then uniform[True, False]
                half = tf_util.constant(value=0.5, dtype='float')
                random_action = tf.random.uniform(shape=shape, dtype=float_dtype) < half
                is_random = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                actions[name] = tf.where(condition=is_random, x=random_action, y=actions[name])

            elif spec.type == 'int' and spec.num_values is not None:
                if self.config.enable_int_action_masking:
                    # Masked action: if uniform[0, 1] < exploration, then uniform[unmasked]
                    # (Similar code as for RandomModel.core_act)
                    mask = auxiliaries[name + '_mask']
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
                    random_action = tf.gather(params=choices, indices=(masked_offset + random_offset))
                    is_random = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                    actions[name] = tf.where(condition=is_random, x=random_action, y=actions[name])

                else:
                    # Bounded int action: if uniform[0, 1] < exploration, then uniform[num_values]
                    random_action = tf.random.uniform(
                        shape=shape, maxval=spec.num_values, dtype=spec.tf_type()
                    )
                    is_random = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                    actions[name] = tf.where(condition=is_random, x=random_action, y=actions[name])

            else:
                # Int/float action: action + normal[0, exploration]
                actions[name] += tf.random.normal(shape=shape, dtype=spec.tf_type()) * exploration

                # Clip action if left-/right-bounded
                if spec.min_value is not None:
                    actions[name] = tf.math.maximum(x=actions[name], y=spec.min_value)
                if spec.max_value is not None:
                    actions[name] = tf.math.minimum(x=actions[name], y=spec.max_value)

        return actions

    def get_variable(self, variable):
        if not variable.startswith(self.name):
            variable = util.join_scopes(self.name, variable)
        fetches = variable + '-output:0'
        return self.monitored_session.run(fetches=fetches)

    def assign_variable(self, variable, value):
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

    def summarize(self, summary, value, step=None):
        fetches = util.join_scopes(self.name, summary, 'write_summary', 'Const:0')
        feed_dict = {util.join_scopes(self.name, 'summarize-input:0'): value}
        if step is not None:
            feed_dict[util.join_scopes(self.name, 'summarize-step-input:0')] = step
        self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)

    def save(self, directory, filename, format, append=None, no_act_pb=False):
        path = os.path.join(directory, filename)

        if append is not None:
            append = self.root.graph.get_collection(name=util.join_scopes(self.name, append))[0]

        if format == 'tensorflow':
            if self.summarizer_spec is not None:
                self.monitored_session.run(fetches=self.summarizer_flush)
            saver_path = self.saver.save(
                sess=self.session, save_path=path, global_step=append,
                # latest_filename=None,  # Defaults to 'checkpoint'.
                meta_graph_suffix='meta', write_meta_graph=True, write_state=True
            )
            assert saver_path.startswith(path)
            path = saver_path

            if not no_act_pb:
                graph_def = self.graph.as_graph_def()

                # freeze_graph clear_devices option
                for node in graph_def.node:
                    node.device = ''

                graph_def = tf.compat.v1.graph_util.remove_training_nodes(input_graph=graph_def)
                output_node_names = [
                    self.name + '.independent_act/' + name + '-output'
                    for name in self.output_tensors['independent_act']
                ]
                # implies tf.compat.v1.graph_util.extract_sub_graph
                graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                    sess=self.monitored_session, input_graph_def=graph_def,
                    output_node_names=output_node_names
                )
                graph_path = tf.io.write_graph(
                    graph_or_graph_def=graph_def, logdir=directory,
                    name=(os.path.split(path)[1] + '.pb'), as_text=False
                )
                assert graph_path == path + '.pb'

        elif format == 'numpy':
            if append is not None:
                append = self.monitored_session.run(fetches=append)
                path += '-' + str(append)
            path += '.npz'
            variables = dict()
            for variable in self.saved_variables:
                name = variable.name[len(self.name) + 1: -2]
                variables[name] = self.get_variable(variable=name)
            np.savez(file=path, **variables)

        elif format == 'hdf5':
            if append is not None:
                append = self.monitored_session.run(fetches=append)
                path += '-' + str(append)
            path += '.hdf5'
            with h5py.File(name=path, mode='w') as filehandle:
                for variable in self.saved_variables:
                    name = variable.name[len(self.name) + 1: -2]
                    filehandle.create_dataset(name=name, data=self.get_variable(variable=name))

        else:
            assert False

        return path

    def restore(self, directory, filename, format):
        path = os.path.join(directory, filename)

        if format == 'tensorflow':
            self.saver.restore(sess=self.session, save_path=path)

        elif format == 'numpy':
            variables = np.load(file=(path + '.npz'))
            for variable in self.saved_variables:
                name = variable.name[len(self.name) + 1: -2]
                self.assign_variable(variable=name, value=variables[name])

        elif format == 'hdf5':
            if os.path.isfile(path + '.hdf5'):
                path = path + '.hdf5'
            else:
                path = path + '.h5'
            with h5py.File(name=path, mode='r') as filehandle:
                for variable in self.saved_variables:
                    name = variable.name[len(self.name) + 1: -2]
                    self.assign_variable(variable=name, value=filehandle[name])

        else:
            assert False

        fetches = (
            self.global_tensor(name='timesteps'), self.global_tensor(name='episodes'),
            self.global_tensor(name='updates')
        )
        return self.monitored_session.run(fetches=fetches)
