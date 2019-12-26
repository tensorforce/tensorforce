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

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import memory_modules, Module, optimizer_modules, parameter_modules
from tensorforce.core.estimators import Estimator
from tensorforce.core.models import Model
from tensorforce.core.objectives import objective_modules
from tensorforce.core.policies import policy_modules


class TensorforceModel(Model):

    def __init__(
        self,
        # Model
        name, device, parallel_interactions, buffer_observe, seed, execution, saver, summarizer,
        config, states, actions, preprocessing, exploration, variable_noise, l2_regularization,
        # TensorforceModel
        policy, memory, update, optimizer, objective, reward_estimation, baseline_policy,
        baseline_optimizer, baseline_objective, entropy_regularization
    ):
        # Policy internals specification
        policy_cls, first_arg, kwargs = Module.get_module_class_and_kwargs(
            name='policy', module=policy, modules=policy_modules, states_spec=states,
            actions_spec=actions
        )
        if first_arg is None:
            internals = policy_cls.internals_spec(name='policy', **kwargs)
        else:
            internals = policy_cls.internals_spec(first_arg, name='policy', **kwargs)
        if any(name.startswith('baseline-') for name in internals):
            raise TensorforceError.unexpected()

        # Baseline internals specification
        if baseline_policy is None:
            pass
        else:
            baseline_cls, first_arg, kwargs = Module.get_module_class_and_kwargs(
                name='baseline', module=baseline_policy, modules=policy_modules,
                states_spec=states, actions_spec=actions
            )
            if first_arg is None:
                baseline_internals = baseline_cls.internals_spec(name='baseline', **kwargs)
            else:
                baseline_internals = baseline_cls.internals_spec(
                    first_arg, name='baseline', **kwargs
                )
            for name, spec in baseline_internals.items():
                if name in internals:
                    raise TensorforceError(
                        "Name overlap between policy and baseline internals: {}.".format(name)
                    )
                internals[name] = spec

        super().__init__(
            # Model
            name=name, device=device, parallel_interactions=parallel_interactions,
            buffer_observe=buffer_observe, seed=seed, execution=execution, saver=saver,
            summarizer=summarizer, config=config, states=states, internals=internals,
            actions=actions, preprocessing=preprocessing, exploration=exploration,
            variable_noise=variable_noise, l2_regularization=l2_regularization
        )

        # Policy
        self.policy = self.add_module(
            name='policy', module=policy, modules=policy_modules, states_spec=self.states_spec,
            actions_spec=self.actions_spec
        )

        # Memory
        self.memory = self.add_module(
            name='memory', module=memory, modules=memory_modules, is_trainable=False,
            values_spec=self.values_spec
        )

        # Update mode
        if not all(key in ('batch_size', 'frequency', 'start', 'unit') for key in update):
            raise TensorforceError.value(name='update', value=list(update))
        # update: unit
        elif 'unit' not in update:
            raise TensorforceError.required(name='update', value='unit')
        elif update['unit'] not in ('timesteps', 'episodes'):
            raise TensorforceError.value(
                name='update', argument='unit', value=update['unit']
            )
        # update: batch_size
        elif 'batch_size' not in update:
            raise TensorforceError.required(name='update', value='batch_size')

        self.update_unit = update['unit']
        self.update_batch_size = self.add_module(
            name='update-batch-size', module=update['batch_size'], modules=parameter_modules,
            is_trainable=False, dtype='long'
        )
        if 'frequency' in update and update['frequency'] == 'never':
            self.update_frequency = 'never'
        else:
            self.update_frequency = self.add_module(
                name='update-frequency', module=update.get('frequency', update['batch_size']),
                modules=parameter_modules, is_trainable=False, dtype='long'
            )
            self.update_start = self.add_module(
                name='update-start', module=update.get('start', 0), modules=parameter_modules,
                is_trainable=False, dtype='long'
            )

        # Optimizer
        self.optimizer = self.add_module(
            name='optimizer', module=optimizer, modules=optimizer_modules, is_trainable=False
        )

        # Objective
        self.objective = self.add_module(
            name='objective', module=objective, modules=objective_modules, is_trainable=False
        )

        # Estimator
        if not all(key in (
            'capacity', 'discount', 'estimate_actions', 'estimate_advantage', 'estimate_horizon',
            'estimate_terminal', 'horizon'
        ) for key in reward_estimation):
            raise TensorforceError.value(name='reward_estimation', value=list(reward_estimation))
        if baseline_policy is None and baseline_optimizer is None and baseline_objective is None:
            estimate_horizon = False
        else:
            estimate_horizon = 'late'
        self.estimator = self.add_module(
            name='estimator', module=Estimator, is_trainable=False, is_saved=False,
            values_spec=self.values_spec, horizon=reward_estimation['horizon'],
            discount=reward_estimation.get('discount', 1.0),
            estimate_horizon=reward_estimation.get('estimate_horizon', estimate_horizon),
            estimate_actions=reward_estimation.get('estimate_actions', False),
            estimate_terminal=reward_estimation.get('estimate_terminal', False),
            estimate_advantage=reward_estimation.get('estimate_advantage', False),
            capacity=reward_estimation['capacity']
        )

        # Baseline
        if (baseline_policy is not None or baseline_objective is not None) and \
                (baseline_optimizer is None or isinstance(baseline_optimizer, float)):
            # since otherwise not part of training
            assert self.estimator.estimate_advantage or baseline_objective is not None
            is_trainable = True
        else:
            is_trainable = False
        if baseline_policy is None:
            self.baseline_policy = self.policy
        else:
            self.baseline_policy = self.add_module(
                name='baseline', module=baseline_policy, modules=policy_modules,
                is_trainable=is_trainable, is_subscope=True, states_spec=self.states_spec,
                actions_spec=self.actions_spec
            )

        # Baseline optimizer
        if baseline_optimizer is None:
            self.baseline_optimizer = None
            self.baseline_loss_weight = 1.0
        elif isinstance(baseline_optimizer, float):
            self.baseline_optimizer = None
            self.baseline_loss_weight = baseline_optimizer
        else:
            self.baseline_optimizer = self.add_module(
                name='baseline-optimizer', module=baseline_optimizer, modules=optimizer_modules,
                is_trainable=False, is_subscope=True
            )

        # Baseline objective
        if baseline_objective is None:
            self.baseline_objective = None
        else:
            self.baseline_objective = self.add_module(
                name='baseline-objective', module=baseline_objective, modules=objective_modules,
                is_trainable=False, is_subscope=True
            )

        # Entropy regularization
        entropy_regularization = 0.0 if entropy_regularization is None else entropy_regularization
        self.entropy_regularization = self.add_module(
            name='entropy-regularization', module=entropy_regularization,
            modules=parameter_modules, is_trainable=False, dtype='float'
        )

        # Internals initialization
        self.internals_init.update(self.policy.internals_init())
        self.internals_init.update(self.baseline_policy.internals_init())
        if any(internal_init is None for internal_init in self.internals_init.values()):
            raise TensorforceError.unexpected()

        # Register global tensors
        Module.register_tensor(name='update', spec=dict(type='long', shape=()), batched=False)
        Module.register_tensor(
            name='optimization', spec=dict(type='bool', shape=()), batched=False
        )
        Module.register_tensor(
            name='dependency_starts', spec=dict(type='long', shape=()), batched=True
        )
        Module.register_tensor(
            name='dependency_lengths', spec=dict(type='long', shape=()), batched=True
        )

    def tf_initialize(self):
        super().tf_initialize()

        # Internals
        self.internals_input = OrderedDict()
        for name, internal_spec in self.internals_spec.items():
            self.internals_input[name] = self.add_placeholder(
                name=name, dtype=internal_spec['type'], shape=internal_spec['shape'], batched=True
            )

        # Actions
        self.actions_input = OrderedDict()
        for name, action_spec in self.actions_spec.items():
            self.actions_input[name] = self.add_placeholder(
                name=name, dtype=action_spec['type'], shape=action_spec['shape'], batched=True
            )

    def api_experience(self):
        # Inputs
        states = self.states_input
        internals = self.internals_input
        auxiliaries = self.auxiliaries_input
        actions = self.actions_input
        terminal = self.terminal_input
        reward = self.reward_input

        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        batch_size = tf.shape(input=terminal)[:1]

        # Assertions
        assertions = list()
        # terminal: type and shape
        tf.debugging.assert_type(tensor=terminal, tf_type=util.tf_dtype(dtype='long'))
        assertions.append(tf.debugging.assert_rank(x=terminal, rank=1))
        # reward: type and shape
        tf.debugging.assert_type(tensor=reward, tf_type=util.tf_dtype(dtype='float'))
        assertions.append(tf.debugging.assert_rank(x=reward, rank=1))
        # shape of terminal equals shape of reward
        assertions.append(tf.debugging.assert_equal(
            x=tf.shape(input=terminal), y=tf.shape(input=reward)
        ))
        # buffer index is zero
        assertions.append(tf.debugging.assert_equal(
            x=tf.math.reduce_sum(input_tensor=self.buffer_index, axis=0),
            y=tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        ))
        # at most one terminal
        assertions.append(tf.debugging.assert_less_equal(
            x=tf.math.count_nonzero(input=terminal, dtype=util.tf_dtype(dtype='long')),
            y=tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        ))
        # if terminal, last timestep in batch
        assertions.append(tf.debugging.assert_equal(
            x=tf.math.reduce_any(input_tensor=tf.math.greater(x=terminal, y=zero)),
            y=tf.math.greater(x=terminal[-1], y=zero)
        ))
        # states: type and shape
        for name, spec in self.states_spec.items():
            tf.debugging.assert_type(
                tensor=states[name], tf_type=util.tf_dtype(dtype=spec['type'])
            )
            shape = self.unprocessed_state_shape.get(name, spec['shape'])
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf.shape(input=states[name], out_type=tf.int32),
                    y=tf.concat(
                        values=(batch_size, tf.constant(value=shape, dtype=tf.int32)), axis=0
                    )
                )
            )
        # internals: type and shape
        for name, spec in self.internals_spec.items():
            tf.debugging.assert_type(
                tensor=internals[name], tf_type=util.tf_dtype(dtype=spec['type'])
            )
            shape = spec['shape']
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf.shape(input=internals[name], out_type=tf.int32),
                    y=tf.concat(
                        values=(batch_size, tf.constant(value=shape, dtype=tf.int32)), axis=0
                    )
                )
            )
        # action_masks: type and shape
        for name, spec in self.actions_spec.items():
            if spec['type'] == 'int':
                name = name + '_mask'
                tf.debugging.assert_type(
                    tensor=auxiliaries[name], tf_type=util.tf_dtype(dtype='bool')
                )
                shape = spec['shape'] + (spec['num_values'],)
                assertions.append(
                    tf.debugging.assert_equal(
                        x=tf.shape(input=auxiliaries[name], out_type=tf.int32),
                        y=tf.concat(
                            values=(batch_size, tf.constant(value=shape, dtype=tf.int32)), axis=0
                        )
                    )
                )
        # actions: type and shape
        for name, spec in self.actions_spec.items():
            tf.debugging.assert_type(
                tensor=actions[name], tf_type=util.tf_dtype(dtype=spec['type'])
            )
            shape = spec['shape']
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf.shape(input=actions[name], out_type=tf.int32),
                    y=tf.concat(
                        values=(batch_size, tf.constant(value=shape, dtype=tf.int32)), axis=0
                    )
                )
            )

        # Set global tensors
        Module.update_tensors(
            deterministic=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')),
            independent=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')),
            optimization=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool')),
            timestep=self.global_timestep, episode=self.global_episode, update=self.global_update
        )

        with tf.control_dependencies(control_inputs=assertions):
            # Core experience: retrieve experience operation
            experienced = self.core_experience(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                terminal=terminal, reward=reward
            )

        with tf.control_dependencies(control_inputs=(experienced,)):
            # Function-level identity operation for retrieval (plus enforce dependency)
            timestep = util.identity_operation(
                x=self.global_timestep, operation_name='timestep-output'
            )
            episode = util.identity_operation(
                x=self.global_episode, operation_name='episode-output'
            )
            update = util.identity_operation(
                x=self.global_update, operation_name='update-output'
            )

        return timestep, episode, update

    def api_update(self):
        # Set global tensors
        Module.update_tensors(
            deterministic=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')),
            independent=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool')),
            optimization=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')),
            timestep=self.global_timestep, episode=self.global_episode, update=self.global_update
        )

        # Core update: retrieve update operation
        updated = self.core_update()

        with tf.control_dependencies(control_inputs=(updated,)):
            # Function-level identity operation for retrieval (plus enforce dependency)
            timestep = util.identity_operation(
                x=self.global_timestep, operation_name='timestep-output'
            )
            episode = util.identity_operation(
                x=self.global_episode, operation_name='episode-output'
            )
            update = util.identity_operation(
                x=self.global_update, operation_name='update-output'
            )

        return timestep, episode, update

    def tf_core_act(self, states, internals, auxiliaries):
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))

        # Dependency horizon
        dependency_horizon = self.policy.dependency_horizon(is_optimization=False)
        dependency_horizon = tf.math.maximum(
            x=dependency_horizon, y=self.baseline_policy.dependency_horizon(is_optimization=False)
        )

        # TODO: handle arbitrary non-optimization horizons!
        assertion = tf.debugging.assert_equal(x=dependency_horizon, y=zero)
        with tf.control_dependencies(control_inputs=(assertion,)):
            some_state = next(iter(states.values()))
            if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
                batch_size = tf.shape(input=some_state, out_type=util.tf_dtype(dtype='long'))[0]
            else:
                batch_size = tf.dtypes.cast(
                    x=tf.shape(input=some_state)[0], dtype=util.tf_dtype(dtype='long')
                )
            starts = tf.range(start=batch_size, dtype=util.tf_dtype(dtype='long'))
            lengths = tf.ones(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
            Module.update_tensors(dependency_starts=starts, dependency_lengths=lengths)

        # Policy act
        actions, next_internals = self.policy.act(
            states=states, internals=internals, auxiliaries=auxiliaries, return_internals=True
        )

        if any(name not in next_internals for name in internals):
            # Baseline policy act to retrieve next internals
            _, baseline_internals = self.baseline_policy.act(
                states=states, internals=internals, auxiliaries=auxiliaries, return_internals=True
            )
            assert all(name not in next_internals for name in baseline_internals)
            next_internals.update(baseline_internals)

        return actions, next_internals

    def tf_core_observe(self, states, internals, auxiliaries, actions, terminal, reward):
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))

        # Experience
        experienced = self.core_experience(
            states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
            terminal=terminal, reward=reward
        )

        # If no periodic update
        if self.update_frequency == 'never':
            return experienced

        # Periodic update
        with tf.control_dependencies(control_inputs=(experienced,)):
            batch_size = self.update_batch_size.value()
            frequency = self.update_frequency.value()
            start = self.update_start.value()

            if self.update_unit == 'timesteps':
                # Timestep-based batch
                one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
                past_horizon = self.policy.dependency_horizon(is_optimization=True)
                past_horizon = tf.math.maximum(
                    x=past_horizon, y=self.baseline_policy.dependency_horizon(is_optimization=True)
                )
                future_horizon = self.estimator.horizon.value() + one
                start = tf.math.maximum(x=start, y=(batch_size + past_horizon + future_horizon))
                timestep = Module.retrieve_tensor(name='timestep')
                timestep = timestep - self.estimator.capacity
                is_frequency = tf.math.equal(x=tf.math.mod(x=timestep, y=frequency), y=zero)
                at_least_start = tf.math.greater_equal(x=timestep, y=start)

            elif self.update_unit == 'episodes':
                # Episode-based batch
                start = tf.math.maximum(x=start, y=batch_size)
                episode = Module.retrieve_tensor(name='episode')
                is_frequency = tf.math.equal(x=tf.math.mod(x=episode, y=frequency), y=zero)
                # Only update once per episode increment
                terminal = tf.concat(values=((zero,), terminal), axis=0)
                is_frequency = tf.math.logical_and(x=is_frequency, y=(terminal[-1] > zero))
                at_least_start = tf.math.greater_equal(x=episode, y=start)

            def perform_update():
                return self.core_update()

            is_updated = self.cond(
                pred=tf.math.logical_and(x=is_frequency, y=at_least_start), true_fn=perform_update,
                false_fn=util.no_operation
            )

        return is_updated

    def tf_core_experience(self, states, internals, auxiliaries, actions, terminal, reward):
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))

        # Enqueue experience for early reward estimation
        any_overwritten, overwritten_values = self.estimator.enqueue(
            baseline=self.baseline_policy, states=states, internals=internals,
            auxiliaries=auxiliaries, actions=actions, terminal=terminal, reward=reward
        )

        # If terminal, store remaining values in memory

        def true_fn():
            reset_values = self.estimator.reset(baseline=self.baseline_policy)

            new_overwritten_values = OrderedDict()
            for name, value1, value2 in util.zip_items(overwritten_values, reset_values):
                if util.is_nested(name=name):
                    new_overwritten_values[name] = OrderedDict()
                    for inner_name, value1, value2 in util.zip_items(value1, value2):
                        new_overwritten_values[name][inner_name] = tf.concat(
                            values=(value1, value2), axis=0
                        )
                else:
                    new_overwritten_values[name] = tf.concat(values=(value1, value2), axis=0)
            return new_overwritten_values

        def false_fn():
            return overwritten_values

        with tf.control_dependencies(control_inputs=util.flatten(xs=overwritten_values)):
            values = self.cond(pred=(terminal[-1] > zero), true_fn=true_fn, false_fn=false_fn)

        # If any, store overwritten values
        def store():
            return self.memory.enqueue(**values)

        terminal = values['terminal']
        if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
            num_values = tf.shape(input=terminal, out_type=util.tf_dtype(dtype='long'))[0]
        else:
            num_values = tf.dtypes.cast(
                x=tf.shape(input=terminal)[0], dtype=util.tf_dtype(dtype='long')
            )

        stored = self.cond(pred=(num_values > zero), true_fn=store, false_fn=util.no_operation)

        return stored

    def tf_core_update(self):
        Module.update_tensor(name='update', tensor=self.global_update)

        true = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))

        # Retrieve batch
        batch_size = self.update_batch_size.value()
        if self.update_unit == 'timesteps':
            # Timestep-based batch
            # Dependency horizon
            past_horizon = self.policy.dependency_horizon(is_optimization=True)
            past_horizon = tf.math.maximum(
                x=past_horizon, y=self.baseline_policy.dependency_horizon(is_optimization=True)
            )
            future_horizon = self.estimator.horizon.value() + one
            indices = self.memory.retrieve_timesteps(
                n=batch_size, past_padding=past_horizon, future_padding=future_horizon
            )
        elif self.update_unit == 'episodes':
            # Episode-based batch
            indices = self.memory.retrieve_episodes(n=batch_size)

        # Optimization
        optimized = self.optimize(indices=indices)

        # Increment update
        with tf.control_dependencies(control_inputs=(optimized,)):
            assignment = self.global_update.assign_add(delta=one, read_value=False)

        with tf.control_dependencies(control_inputs=(assignment,)):
            return util.identity_operation(x=true)

    def tf_optimize(self, indices):
        # Baseline optimization
        if self.baseline_optimizer is not None:
            optimized = self.optimize_baseline(indices=indices)
            dependencies = (optimized,)
        else:
            dependencies = (indices,)

        # Reward estimation
        with tf.control_dependencies(control_inputs=dependencies):
            reward = self.memory.retrieve(indices=indices, values='reward')
            reward = self.estimator.complete(
                baseline=self.baseline_policy, memory=self.memory, indices=indices, reward=reward
            )
            reward = self.add_summary(
                label=('empirical-reward', 'rewards'), name='empirical-reward', tensor=reward
            )
            reward = self.estimator.estimate(
                baseline=self.baseline_policy, memory=self.memory, indices=indices, reward=reward
            )
            reward = self.add_summary(
                label=('estimated-reward', 'rewards'), name='estimated-reward', tensor=reward
            )

        # Stop gradients of estimated rewards if separate baseline optimization
        if self.baseline_optimizer is not None or self.baseline_objective is not None:
            reward = tf.stop_gradient(input=reward)

        # Retrieve states, internals and actions
        dependency_horizon = self.policy.dependency_horizon(is_optimization=True)
        if self.baseline_optimizer is None:
            assertion = tf.debugging.assert_equal(
                x=dependency_horizon,
                y=self.baseline_policy.dependency_horizon(is_optimization=True)
            )
        else:
            assertion = dependency_horizon

        with tf.control_dependencies(control_inputs=(assertion,)):
            # horizon change: see timestep-based batch sampling
            starts, lengths, states, internals = self.memory.predecessors(
                indices=indices, horizon=dependency_horizon, sequence_values='states',
                initial_values='internals'
            )
            Module.update_tensors(dependency_starts=starts, dependency_lengths=lengths)
            auxiliaries, actions = self.memory.retrieve(
                indices=indices, values=('auxiliaries', 'actions')
            )

        # Optimizer arguments
        variables = self.get_variables(only_trainable=True)

        arguments = dict(
            states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
            reward=reward
        )

        fn_loss = self.total_loss

        def fn_kl_divergence(states, internals, auxiliaries, actions, reward, other=None):
            kl_divergence = self.policy.kl_divergence(
                states=states, internals=internals, auxiliaries=auxiliaries, other=other
            )
            if self.baseline_optimizer is None and self.baseline_objective is not None:
                kl_divergence += self.baseline_policy.kl_divergence(
                    states=states, internals=internals, auxiliaries=auxiliaries, other=other
                )
            return kl_divergence

        if self.global_model is None:
            global_variables = None
        else:
            global_variables = self.global_model.get_variables(only_trainable=True)

        kwargs = self.objective.optimizer_arguments(
            policy=self.policy, baseline=self.baseline_policy
        )

        if self.baseline_optimizer is None and self.baseline_objective is not None:
            util.deep_disjoint_update(
                target=kwargs,
                source=self.baseline_objective.optimizer_arguments(policy=self.baseline_policy)
            )

        dependencies = util.flatten(xs=arguments)

        # KL divergence before
        if self.is_summary_logged(
            label=('kl-divergence', 'action-kl-divergences', 'kl-divergences')
        ):
            with tf.control_dependencies(control_inputs=dependencies):
                kldiv_reference = self.policy.kldiv_reference(
                    states=states, internals=internals, auxiliaries=auxiliaries
                )
                dependencies = util.flatten(xs=kldiv_reference)

        # Optimization
        with tf.control_dependencies(control_inputs=dependencies):
            optimized = self.optimizer.minimize(
                variables=variables, arguments=arguments, fn_loss=fn_loss,
                fn_kl_divergence=fn_kl_divergence, global_variables=global_variables, **kwargs
            )

        with tf.control_dependencies(control_inputs=(optimized,)):
            # Loss summaries
            if self.is_summary_logged(label=('loss', 'objective-loss', 'losses')):
                objective_loss = self.objective.loss_per_instance(policy=self.policy, **arguments)
                objective_loss = tf.math.reduce_mean(input_tensor=objective_loss, axis=0)
            if self.is_summary_logged(label=('objective-loss', 'losses')):
                optimized = self.add_summary(
                    label=('objective-loss', 'losses'), name='objective-loss',
                    tensor=objective_loss, pass_tensors=optimized
                )
            if self.is_summary_logged(label=('loss', 'regularization-loss', 'losses')):
                regularization_loss = self.regularize(
                    states=states, internals=internals, auxiliaries=auxiliaries
                )
            if self.is_summary_logged(label=('regularization-loss', 'losses')):
                optimized = self.add_summary(
                    label=('regularization-loss', 'losses'), name='regularization-loss',
                    tensor=regularization_loss, pass_tensors=optimized
                )
            if self.is_summary_logged(label=('loss', 'losses')):
                loss = objective_loss + regularization_loss
            if self.baseline_optimizer is None and self.baseline_objective is not None:
                if self.is_summary_logged(label=('loss', 'baseline-objective-loss', 'losses')):
                    if self.baseline_objective is None:
                        baseline_objective_loss = self.objective.loss_per_instance(
                            policy=self.baseline_policy, **arguments
                        )
                    else:
                        baseline_objective_loss = self.baseline_objective.loss_per_instance(
                            policy=self.baseline_policy, **arguments
                        )
                    baseline_objective_loss = tf.math.reduce_mean(
                        input_tensor=baseline_objective_loss, axis=0
                    )
                if self.is_summary_logged(label=('baseline-objective-loss', 'losses')):
                    optimized = self.add_summary(
                        label=('baseline-objective-loss', 'losses'),
                        name='baseline-objective-loss', tensor=baseline_objective_loss,
                        pass_tensors=optimized
                    )
                if self.is_summary_logged(
                    label=('loss', 'baseline-regularization-loss', 'losses')
                ):
                    baseline_regularization_loss = self.baseline_policy.regularize()
                if self.is_summary_logged(label=('baseline-regularization-loss', 'losses')):
                    optimized = self.add_summary(
                        label=('baseline-regularization-loss', 'losses'),
                        name='baseline-regularization-loss', tensor=baseline_regularization_loss,
                        pass_tensors=optimized
                    )
                if self.is_summary_logged(label=('loss', 'baseline-loss', 'losses')):
                    baseline_loss = baseline_objective_loss + baseline_regularization_loss
                if self.is_summary_logged(label=('baseline-loss', 'losses')):
                    optimized = self.add_summary(
                        label=('baseline-loss', 'losses'), name='baseline-loss',
                        tensor=baseline_loss, pass_tensors=optimized
                    )
                if self.is_summary_logged(label=('loss', 'losses')):
                    loss += self.baseline_loss_weight * baseline_loss
            if self.is_summary_logged(label=('loss', 'losses')):
                optimized = self.add_summary(
                    label=('loss', 'losses'), name='loss', tensor=loss, pass_tensors=optimized
                )

            # Entropy summaries
            if self.is_summary_logged(label=('entropy', 'action-entropies', 'entropies')):
                entropies = self.policy.entropy(
                    states=states, internals=internals, auxiliaries=auxiliaries,
                    include_per_action=(len(self.actions_spec) > 1)
                )
            if self.is_summary_logged(label=('entropy', 'entropies')):
                if len(self.actions_spec) == 1:
                    optimized = self.add_summary(
                        label=('entropy', 'entropies'), name='entropy', tensor=entropies,
                        pass_tensors=optimized
                    )
                else:
                    optimized = self.add_summary(
                        label=('entropy', 'entropies'), name='entropy', tensor=entropies['*'],
                        pass_tensors=optimized
                    )
            if len(self.actions_spec) > 1 and \
                    self.is_summary_logged(label=('action-entropies', 'entropies')):
                for name in self.actions_spec:
                    optimized = self.add_summary(
                        label=('action-entropies', 'entropies'), name=(name + '-entropy'),
                        tensor=entropies[name], pass_tensors=optimized
                    )

            # KL divergence summaries
            if self.is_summary_logged(
                label=('kl-divergence', 'action-kl-divergences', 'kl-divergences')
            ):
                kl_divergences = self.policy.kl_divergence(
                    states=states, internals=internals, auxiliaries=auxiliaries,
                    other=kldiv_reference, include_per_action=(len(self.actions_spec) > 1)
                )
            if self.is_summary_logged(label=('kl-divergence', 'kl-divergences')):
                if len(self.actions_spec) == 1:
                    optimized = self.add_summary(
                        label=('kl-divergence', 'kl-divergences'), name='kl-divergence',
                        tensor=kl_divergences, pass_tensors=optimized
                    )
                else:
                    optimized = self.add_summary(
                        label=('kl-divergence', 'kl-divergences'), name='kl-divergence',
                        tensor=kl_divergences['*'], pass_tensors=optimized
                    )
            if len(self.actions_spec) > 1 and \
                    self.is_summary_logged(label=('action-kl-divergences', 'kl-divergences')):
                for name in self.actions_spec:
                    optimized = self.add_summary(
                        label=('action-kl-divergences', 'kl-divergences'),
                        name=(name + '-kl-divergence'), tensor=kl_divergences[name],
                        pass_tensors=optimized
                    )

        return optimized

    def tf_total_loss(self, states, internals, auxiliaries, actions, reward, **kwargs):
        # Loss per instance
        loss = self.objective.loss_per_instance(
            policy=self.policy, states=states, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, **kwargs
        )

        # Objective loss
        loss = tf.math.reduce_mean(input_tensor=loss, axis=0)

        # Regularization losses
        loss += self.regularize(
            states=states, internals=internals, auxiliaries=auxiliaries
        )

        # Baseline loss
        if self.baseline_optimizer is None and self.baseline_objective is not None:
            loss += self.baseline_loss_weight * self.baseline_loss(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                reward=reward
            )

        return loss

    def tf_regularize(self, states, internals, auxiliaries):
        regularization_loss = super().tf_regularize(
            states=states, internals=internals, auxiliaries=auxiliaries
        )

        # Entropy regularization
        zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        entropy_regularization = self.entropy_regularization.value()

        def no_entropy_regularization():
            return zero

        def apply_entropy_regularization():
            entropy = self.policy.entropy(
                states=states, internals=internals, auxiliaries=auxiliaries
            )
            entropy = tf.math.reduce_mean(input_tensor=entropy, axis=0)
            return -entropy_regularization * entropy

        skip_entropy_regularization = tf.math.equal(x=entropy_regularization, y=zero)
        regularization_loss += self.cond(
            pred=skip_entropy_regularization, true_fn=no_entropy_regularization,
            false_fn=apply_entropy_regularization
        )

        return regularization_loss

    def tf_optimize_baseline(self, indices):
        # Retrieve states, internals, actions and reward
        dependency_horizon = self.baseline_policy.dependency_horizon(is_optimization=True)
        # horizon change: see timestep-based batch sampling
        starts, lengths, states, internals = self.memory.predecessors(
            indices=indices, horizon=dependency_horizon, sequence_values='states',
            initial_values='internals'
        )
        Module.update_tensors(dependency_starts=starts, dependency_lengths=lengths)
        auxiliaries, actions, reward = self.memory.retrieve(
            indices=indices, values=('auxiliaries', 'actions', 'reward')
        )

        # Reward estimation (separate from main policy, so updated baseline is used there)
        reward = self.memory.retrieve(indices=indices, values='reward')
        reward = self.estimator.complete(
            baseline=self.baseline_policy, memory=self.memory, indices=indices, reward=reward
        )

        # Optimizer arguments
        variables = self.baseline_policy.get_variables(only_trainable=True)

        arguments = dict(
            states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
            reward=reward
        )

        fn_loss = self.baseline_loss

        def fn_kl_divergence(states, internals, auxiliaries, actions, reward, other=None):
            return self.baseline_policy.kl_divergence(
                states=states, internals=internals, auxiliaries=auxiliaries, other=other
            )

        source_variables = self.policy.get_variables(only_trainable=True)

        if self.global_model is None:
            global_variables = None
        else:
            global_variables = self.global_model.baseline_policy.get_variables(only_trainable=True)

        if self.baseline_objective is None:
            kwargs = self.objective.optimizer_arguments(policy=self.baseline_policy)
        else:
            kwargs = self.baseline_objective.optimizer_arguments(policy=self.baseline_policy)

        # Optimization
        optimized = self.baseline_optimizer.minimize(
            variables=variables, arguments=arguments, fn_loss=fn_loss,
            fn_kl_divergence=fn_kl_divergence, source_variables=source_variables,
            global_variables=global_variables, **kwargs
        )

        with tf.control_dependencies(control_inputs=(optimized,)):
            # Loss summaries
            if self.is_summary_logged(
                label=('baseline-loss', 'baseline-objective-loss', 'losses')
            ):
                if self.baseline_objective is None:
                    objective_loss = self.objective.loss_per_instance(
                        policy=self.baseline_policy, **arguments
                    )
                else:
                    objective_loss = self.baseline_objective.loss_per_instance(
                        policy=self.baseline_policy, **arguments
                    )
                objective_loss = tf.math.reduce_mean(input_tensor=objective_loss, axis=0)
            if self.is_summary_logged(label=('baseline-objective-loss', 'losses')):
                optimized = self.add_summary(
                    label=('baseline-objective-loss', 'losses'), name='baseline-objective-loss',
                    tensor=objective_loss, pass_tensors=optimized
                )
            if self.is_summary_logged(
                label=('baseline-loss', 'baseline-regularization-loss', 'losses')
            ):
                regularization_loss = self.baseline_policy.regularize()
            if self.is_summary_logged(label=('baseline-regularization-loss', 'losses')):
                optimized = self.add_summary(
                    label=('baseline-regularization-loss', 'losses'),
                    name='baseline-regularization-loss', tensor=regularization_loss,
                    pass_tensors=optimized
                )
            if self.is_summary_logged(label=('baseline-loss', 'losses')):
                loss = objective_loss + regularization_loss
                optimized = self.add_summary(
                    label=('baseline-loss', 'losses'), name='baseline-loss', tensor=loss,
                    pass_tensors=optimized
                )

        return optimized

    def tf_baseline_loss(self, states, internals, auxiliaries, actions, reward, **kwargs):
        # Loss per instance
        if self.baseline_objective is None:
            loss = self.objective.loss_per_instance(
                policy=self.baseline_policy, states=states, internals=internals,
                auxiliaries=auxiliaries, actions=actions, reward=reward, **kwargs
            )
        else:
            loss = self.baseline_objective.loss_per_instance(
                policy=self.baseline_policy, states=states, internals=internals,
                auxiliaries=auxiliaries, actions=actions, reward=reward, **kwargs
            )

        # Objective loss
        loss = tf.math.reduce_mean(input_tensor=loss, axis=0)

        # Regularization losses
        loss += self.baseline_policy.regularize()

        return loss
