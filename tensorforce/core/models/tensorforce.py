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

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import memory_modules, optimizer_modules, parameter_modules, SignatureDict, \
    TensorDict, TensorSpec, TensorsSpec, tf_function, tf_util
from tensorforce.core.estimators import Estimator
from tensorforce.core.models import Model
from tensorforce.core.networks import Preprocessor
from tensorforce.core.objectives import objective_modules
from tensorforce.core.policies import policy_modules


class TensorforceModel(Model):

    def __init__(
        self, *,
        # Model
        states, actions, preprocessing, exploration, variable_noise, l2_regularization, name,
        device, parallel_interactions, config, saver, summarizer,
        # TensorforceModel
        policy, memory, update, optimizer, objective, reward_estimation, baseline_policy,
        baseline_optimizer, baseline_objective, entropy_regularization, max_episode_timesteps
    ):
        super().__init__(
            # Model
            states=states, actions=actions, preprocessing=preprocessing, exploration=exploration,
            variable_noise=variable_noise, l2_regularization=l2_regularization, name=name,
            device=device, parallel_interactions=parallel_interactions, config=config, saver=saver,
            summarizer=summarizer
        )

        # Return/advantage preprocessing
        if preprocessing is not None:
            if 'return' in preprocessing:
                self.preprocessing['return'] = self.submodule(
                    name=('return_preprocessing'), module=Preprocessor, is_trainable=False,
                    input_spec=self.reward_spec, layers=preprocessing['return']
                )
                if self.preprocessing['return'].get_output_spec() != self.reward_spec:
                    raise TensorforceError.mismatch(
                        name='preprocessing', argument='return output spec',
                        value1=self.preprocessing['return'].get_output_spec(),
                        value2=self.reward_spec
                    )
            if 'advantage' in preprocessing:
                self.preprocessing['advantage'] = self.submodule(
                    name=('advantage_preprocessing'), module=Preprocessor, is_trainable=False,
                    input_spec=self.reward_spec, layers=preprocessing['advantage']
                )
                if self.preprocessing['advantage'].get_output_spec() != self.reward_spec:
                    raise TensorforceError.mismatch(
                        name='preprocessing', argument='advantage output spec',
                        value1=self.preprocessing['advantage'].get_output_spec(),
                        value2=self.reward_spec
                    )

        # Estimator argument check
        if not all(key in (
            'discount', 'estimate_action_values', 'estimate_advantage', 'estimate_horizon',
            'estimate_terminals', 'horizon'
        ) for key in reward_estimation):
            raise TensorforceError.value(
                name='agent', argument='reward_estimation', value=reward_estimation,
                hint='not from {discount,estimate_action_values,estimate_advantage,'
                     'estimate_horizon,estimate_terminals,horizon}'
            )

        # Policy
        # TODO: policy vs value-fn hack
        if isinstance(policy, dict) and 'state_value_mode' in policy:
            kwargs = dict()
        elif isinstance(objective, str) and objective == 'value':
            kwargs = dict()
        elif isinstance(objective, dict) and objective['type'] == 'value' and \
                objective.get('value', 'state') == 'state':
            kwargs = dict()
        elif baseline_policy is not None:
            kwargs = dict(state_value_mode='no-state-value')
        elif not reward_estimation.get('estimate_advantage', False) and (
            reward_estimation.get('estimate_horizon', baseline_objective is not None) is False or
            reward_estimation.get('estimate_action_values', False)
        ):
            kwargs = dict(state_value_mode='no-state-value')
        else:
            kwargs = dict()
        self.policy = self.submodule(
            name='policy', module=policy, modules=policy_modules, states_spec=self.states_spec,
            auxiliaries_spec=self.auxiliaries_spec, actions_spec=self.actions_spec, **kwargs
        )
        self.internals_spec['policy'] = self.policy.internals_spec
        self.internals_init['policy'] = self.policy.internals_init()

        # Update mode
        if not all(key in ('batch_size', 'frequency', 'start', 'unit') for key in update):
            raise TensorforceError.value(
                name='agent', argument='update', value=list(update),
                hint='not from {batch_size,frequency,start,unit}'
            )
        # update: unit
        elif 'unit' not in update:
            raise TensorforceError.required(name='agent', argument='update[unit]')
        elif update['unit'] not in ('timesteps', 'episodes'):
            raise TensorforceError.value(
                name='agent', argument='update[unit]', value=update['unit'],
                hint='not in {timesteps,episodes}'
            )
        # update: batch_size
        elif 'batch_size' not in update:
            raise TensorforceError.required(name='agent', argument='update[batch_size]')

        self.update_unit = update['unit']
        self.update_batch_size = self.submodule(
            name='update_batch_size', module=update['batch_size'], modules=parameter_modules,
            is_trainable=False, dtype='int', min_value=1
        )
        if 'frequency' in update and update['frequency'] == 'never':
            self.update_frequency = None
        else:
            self.update_frequency = self.submodule(
                name='update_frequency', module=update.get('frequency', update['batch_size']),
                modules=parameter_modules, is_trainable=False, dtype='int', min_value=1,
                max_value=max(2, self.update_batch_size.max_value())
            )
            self.update_start = self.submodule(
                name='update_start', module=update.get('start', 0), modules=parameter_modules,
                is_trainable=False, dtype='int', min_value=0
            )

        # Baseline optimization overview:
        # Policy    Objective   Optimizer   Config
        #   n         n           n           default estimate_horizon=False
        #   n         n           f           default estimate_horizon=False
        #   n         n           y           default estimate_horizon=False
        #   n         y           n           main policy, shared loss/kldiv, weighted 1.0
        #   n         y           f           main policy, shared loss/kldiv, weighted
        #   n         y           y           main policy, separate
        #   y         n           n           estimate_in_loss=True, default estimate_advantage=True
        #   y         n           f           shared objective/loss/kldiv, weighted
        #   y         n           y           shared objective
        #   y         y           n           shared loss/kldiv, weighted 1.0, equal horizon
        #   y         y           f           shared loss/kldiv, weighted, equal horizon
        #   y         y           y           separate

        # Defaults
        if baseline_policy is None and baseline_objective is None:
            estimate_horizon = False
        else:
            estimate_horizon = 'late'

        if baseline_policy is not None and baseline_objective is None and \
                baseline_optimizer is None:
            estimate_advantage = True
            self.advantage_in_loss = True
        else:
            estimate_advantage = False
            self.advantage_in_loss = False

        if baseline_optimizer is None and baseline_objective is not None:
            baseline_optimizer = 1.0

        if baseline_optimizer is None or isinstance(baseline_optimizer, float):
            baseline_is_trainable = True
        else:
            baseline_is_trainable = False

        # Baseline
        if baseline_policy is None:
            self.separate_baseline_policy = False
            self.baseline = self.policy
        else:
            self.separate_baseline_policy = True
            # TODO: policy vs value-fn hack
            if isinstance(baseline_policy, dict) and 'state_value_mode' in baseline_policy:
                kwargs = dict()
            elif not reward_estimation.get('estimate_action_values', False):
                if baseline_objective is None:
                    kwargs = dict(state_value_mode='no-distributions')
                elif isinstance(baseline_objective, str) and baseline_objective == 'value':
                    kwargs = dict(state_value_mode='no-distributions')
                elif isinstance(baseline_objective, dict) and \
                        baseline_objective['type'] == 'value' and \
                        baseline_objective.get('value', 'state') == 'state':
                    kwargs = dict(state_value_mode='no-distributions')
                else:
                    kwargs = dict()
            elif reward_estimation.get('estimate_action_values', False) and \
                    not reward_estimation.get('estimate_advantage', False):
                if baseline_objective is None:
                    kwargs = dict(state_value_mode='no-state-value')
                elif isinstance(baseline_objective, str) and baseline_objective != 'value':
                    kwargs = dict(state_value_mode='no-state-value')
                elif isinstance(baseline_objective, dict) and (
                    baseline_objective['type'] != 'value' or
                    baseline_objective.get('value', 'state') != 'state'
                ):
                    kwargs = dict(state_value_mode='no-state-value')
                else:
                    kwargs = dict()
            else:
                kwargs = dict()
            self.baseline = self.submodule(
                name='baseline', module=baseline_policy, modules=policy_modules,
                is_trainable=baseline_is_trainable, states_spec=self.states_spec,
                auxiliaries_spec=self.auxiliaries_spec, actions_spec=self.actions_spec, **kwargs
            )
            self.internals_spec['baseline'] = self.baseline.internals_spec
            self.internals_init['baseline'] = self.baseline.internals_init()

        # Check for name collisions
        for name in self.internals_spec:
            if name in self.value_names:
                raise TensorforceError.exists(name='value name', value=name)
            self.value_names.add(name)

        # Objectives
        self.objective = self.submodule(
            name='policy_objective', module=objective, modules=objective_modules,
            states_spec=self.states_spec, internals_spec=self.internals_spec['policy'],
            auxiliaries_spec=self.auxiliaries_spec, actions_spec=self.actions_spec,
            reward_spec=self.reward_spec
        )
        if self.separate_baseline_policy:
            internals_spec = self.internals_spec['baseline']
        else:
            internals_spec = self.internals_spec['policy']
        if baseline_objective is None:
            self.baseline_objective = None
        else:
            self.baseline_objective = self.submodule(
                name='baseline_objective', module=baseline_objective, modules=objective_modules,
                is_trainable=baseline_is_trainable, states_spec=self.states_spec,
                internals_spec=internals_spec, auxiliaries_spec=self.auxiliaries_spec,
                actions_spec=self.actions_spec, reward_spec=self.reward_spec
            )

        # Optimizers
        if baseline_optimizer is None:
            self.baseline_loss_weight = None
            internals_spec = self.internals_spec
            self.baseline_optimizer = None
        elif isinstance(baseline_optimizer, float):
            self.baseline_loss_weight = self.submodule(
                name='baseline_loss_weight', module=baseline_optimizer, modules=parameter_modules,
                is_trainable=False, dtype='float', min_value=0.0
            )
            internals_spec = self.internals_spec
            self.baseline_optimizer = None
        else:
            self.baseline_loss_weight = None
            internals_spec = self.internals_spec['policy']
            if self.separate_baseline_policy:
                baseline_internals = self.internals_spec['baseline']
            else:
                baseline_internals = self.internals_spec['policy']
            arguments_spec = TensorsSpec(
                states=self.states_spec, horizons=TensorSpec(type='int', shape=(2,)),
                internals=baseline_internals, auxiliaries=self.auxiliaries_spec,
                actions=self.actions_spec, reward=self.reward_spec
            )
            if self.baseline_objective is not None:
                arguments_spec['reference'] = self.baseline_objective.reference_spec()
            self.baseline_optimizer = self.submodule(
                name='baseline_optimizer', module=baseline_optimizer, modules=optimizer_modules,
                is_trainable=False, arguments_spec=arguments_spec
            )
        arguments_spec = TensorsSpec(
            states=self.states_spec, horizons=TensorSpec(type='int', shape=(2,)),
            internals=internals_spec, auxiliaries=self.auxiliaries_spec, actions=self.actions_spec,
            reward=self.reward_spec
        )
        if self.baseline_loss_weight is not None and self.separate_baseline_policy:
            arguments_spec['reference'] = TensorsSpec(
                policy=self.objective.reference_spec(),
                baseline=self.baseline_objective.reference_spec()
            )
        else:
            arguments_spec['reference'] = self.objective.reference_spec()
        self.optimizer = self.submodule(
            name='policy_optimizer', module=optimizer, modules=optimizer_modules,
            arguments_spec=arguments_spec
        )

        # Estimator
        max_past_horizon = self.baseline.max_past_horizon(on_policy=True)
        values_spec = TensorsSpec(
            states=self.states_spec, internals=self.internals_spec,
            auxiliaries=self.auxiliaries_spec, actions=self.actions_spec,
            terminal=self.terminal_spec, reward=self.reward_spec
        )
        self.estimator = self.submodule(
            name='estimator', module=Estimator, is_trainable=False,
            is_saved=False, values_spec=values_spec, horizon=reward_estimation['horizon'],
            discount=reward_estimation.get('discount', 1.0),
            estimate_horizon=reward_estimation.get('estimate_horizon', estimate_horizon),
            estimate_action_values=reward_estimation.get('estimate_action_values', False),
            estimate_terminals=reward_estimation.get('estimate_terminals', False),
            estimate_advantage=reward_estimation.get('estimate_advantage', estimate_advantage),
            min_capacity=self.config.buffer_observe, max_past_horizon=max_past_horizon
        )

        # Memory
        if self.update_unit == 'timesteps':
            max_past_horizon = max(
                self.policy.max_past_horizon(on_policy=False),
                self.baseline.max_past_horizon(on_policy=False)
            )
            min_capacity = self.update_batch_size.max_value() + 1 + max_past_horizon + \
                self.estimator.max_future_horizon()
        elif self.update_unit == 'episodes':
            if max_episode_timesteps is None:
                min_capacity = None
            else:
                min_capacity = (self.update_batch_size.max_value() + 1) * max_episode_timesteps
        else:
            assert False
        # Max enqueue: estimator content plus following batch if terminal
        min_capacity = max(min_capacity, 2 * self.estimator.capacity)

        self.memory = self.submodule(
            name='memory', module=memory, modules=memory_modules, is_trainable=False,
            values_spec=values_spec, min_capacity=min_capacity
        )

        # Entropy regularization
        entropy_regularization = 0.0 if entropy_regularization is None else entropy_regularization
        self.entropy_regularization = self.submodule(
            name='entropy_regularization', module=entropy_regularization,
            modules=parameter_modules, is_trainable=False, dtype='float', min_value=0.0
        )

    def core_initialize(self):
        super().core_initialize()

        # Last update
        self.last_update = self.variable(
            name='last-update', spec=TensorSpec(type='int'), initializer=-1, is_trainable=False,
            is_saved=True
        )

        # Optimizer initialize given variables
        if self.advantage_in_loss:
            self.optimizer.initialize_given_variables(variables=self.trainable_variables)
        else:
            self.optimizer.initialize_given_variables(variables=self.policy.trainable_variables)
        if self.baseline_optimizer is not None:
            self.baseline_optimizer.initialize_given_variables(
                variables=self.baseline.trainable_variables
            )

        # Summaries
        self.register_summary(label='loss', name='losses/policy-objective-loss')
        self.register_summary(label='loss', name='losses/policy-regularization-loss')
        self.register_summary(label='loss', name='losses/policy-loss')
        if self.baseline_optimizer is not None or (
            self.baseline_loss_weight is not None and
            not self.baseline_loss_weight.is_constant(value=0.0)
        ):
            self.register_summary(label='loss', name='losses/baseline-loss')
            if self.separate_baseline_policy:
                self.register_summary(label='loss', name='losses/baseline-objective-loss')
                self.register_summary(label='loss', name='losses/baseline-regularization-loss')

    def initialize_api(self):
        super().initialize_api()

        self.experience(
            states=self.unprocessed_states_spec.empty(batched=True),
            internals=self.internals_spec.empty(batched=True),
            auxiliaries=self.auxiliaries_spec.empty(batched=True),
            actions=self.actions_spec.empty(batched=True),
            terminal=self.terminal_spec.empty(batched=True),
            reward=self.reward_spec.empty(batched=True)
        )
        # TODO: Not possible as it tries to retrieve experiences from memory
        # self.update()

    def input_signature(self, *, function):
        if function == 'baseline_loss':
            if self.baseline_objective is None:
                return SignatureDict(
                    states=self.states_spec.signature(batched=True),
                    horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                    internals=(
                        self.internals_spec['baseline'].signature(batched=True)
                        if self.separate_baseline_policy else
                        self.internals_spec['policy'].signature(batched=True)
                    ),
                    auxiliaries=self.auxiliaries_spec.signature(batched=True),
                    actions=self.actions_spec.signature(batched=True),
                    reward=self.reward_spec.signature(batched=True)
                )
            else:
                return SignatureDict(
                    states=self.states_spec.signature(batched=True),
                    horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                    internals=(
                        self.internals_spec['baseline'].signature(batched=True)
                        if self.separate_baseline_policy else
                        self.internals_spec['policy'].signature(batched=True)
                    ),
                    auxiliaries=self.auxiliaries_spec.signature(batched=True),
                    actions=self.actions_spec.signature(batched=True),
                    reward=self.reward_spec.signature(batched=True),
                    reference=self.baseline_objective.reference_spec().signature(batched=True)
                )

        elif function == 'core_experience':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True),
                terminal=self.terminal_spec.signature(batched=True),
                reward=self.reward_spec.signature(batched=True)
            )

        elif function == 'core_update':
            return SignatureDict()

        elif function == 'experience':
            return SignatureDict(
                states=self.unprocessed_states_spec.signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True),
                terminal=self.terminal_spec.signature(batched=True),
                reward=self.reward_spec.signature(batched=True)
            )

        elif function == 'loss':
            if self.baseline_loss_weight is not None and self.separate_baseline_policy:
                return SignatureDict(
                    states=self.states_spec.signature(batched=True),
                    horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                    internals=self.internals_spec.signature(batched=True),
                    auxiliaries=self.auxiliaries_spec.signature(batched=True),
                    actions=self.actions_spec.signature(batched=True),
                    reward=self.reward_spec.signature(batched=True),
                    reference=SignatureDict(
                        policy=self.objective.reference_spec().signature(batched=True),
                        baseline=self.baseline_objective.reference_spec().signature(batched=True)
                    )
                )
            elif self.baseline_optimizer is None:
                return SignatureDict(
                    states=self.states_spec.signature(batched=True),
                    horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                    internals=self.internals_spec.signature(batched=True),
                    auxiliaries=self.auxiliaries_spec.signature(batched=True),
                    actions=self.actions_spec.signature(batched=True),
                    reward=self.reward_spec.signature(batched=True),
                    reference=self.objective.reference_spec().signature(batched=True)
                )
            else:
                return SignatureDict(
                    states=self.states_spec.signature(batched=True),
                    horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                    internals=self.internals_spec['policy'].signature(batched=True),
                    auxiliaries=self.auxiliaries_spec.signature(batched=True),
                    actions=self.actions_spec.signature(batched=True),
                    reward=self.reward_spec.signature(batched=True),
                    reference=self.objective.reference_spec().signature(batched=True)
                )

        elif function == 'regularize':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec['policy'].signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        elif function == 'update':
            return SignatureDict()

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=0)
    def reset(self):
        reset = self.estimator.reset()

        with tf.control_dependencies(control_inputs=(reset,)):
            timestep, episode, update = super().reset()

        return timestep, episode, update

    @tf_function(num_args=6)
    def experience(self, *, states, internals, auxiliaries, actions, terminal, reward):
        true = tf_util.constant(value=True, dtype='bool')
        zero = tf_util.constant(value=0, dtype='int')
        batch_size = tf_util.cast(x=tf.shape(input=terminal)[0], dtype='int')

        # Input assertions
        assertions = self.unprocessed_states_spec.tf_assert(
            x=states, batch_size=batch_size,
            message='Agent.experience: invalid {issue} for {name} state input.'
        )
        assertions.extend(self.internals_spec.tf_assert(
            x=internals, batch_size=batch_size,
            message='Agent.experience: invalid {issue} for {name} internal input.'
        ))
        assertions.extend(self.auxiliaries_spec.tf_assert(
            x=auxiliaries, batch_size=batch_size,
            message='Agent.experience: invalid {issue} for {name} input.'
        ))
        assertions.extend(self.actions_spec.tf_assert(
            x=actions, batch_size=batch_size,
            message='Agent.experience: invalid {issue} for {name} action input.'
        ))
        assertions.extend(self.terminal_spec.tf_assert(
            x=terminal, batch_size=batch_size,
            message='Agent.experience: invalid {issue} for terminal input.'
        ))
        assertions.extend(self.reward_spec.tf_assert(
            x=reward, batch_size=batch_size,
            message='Agent.experience: invalid {issue} for reward input.'
        ))
        # Mask assertions
        if self.config.enable_int_action_masking:
            for name, spec in self.actions_spec.items():
                if spec.type == 'int' and spec.num_values is not None:
                    is_valid = tf.reduce_all(input_tensor=tf.gather(
                        params=auxiliaries[name]['mask'],
                        indices=tf.expand_dims(input=actions[name], axis=(spec.rank + 1)),
                        batch_dims=(spec.rank + 1)
                    ))
                    assertions.append(tf.debugging.assert_equal(
                        x=is_valid, y=true, message="Agent.experience: invalid action / mask."
                    ))
        # Assertion: buffer indices is zero
        assertions.append(tf.debugging.assert_equal(
            x=tf.math.reduce_sum(input_tensor=self.buffer_index, axis=0), y=zero,
            message="Agent.experience: cannot be called mid-episode."
        ))
        # Assertion: at most one terminal
        assertions.append(tf.debugging.assert_less_equal(
            x=tf.math.count_nonzero(input=terminal, dtype=tf_util.get_dtype(type='int')),
            y=tf_util.constant(value=1, dtype='int'),
            message="Agent.experience: input contains more than one terminal."
        ))
        # Assertion: if terminal, last timestep in batch
        assertions.append(tf.debugging.assert_equal(
            x=tf.math.reduce_any(
                input_tensor=tf.math.greater(x=tf.concat(values=([zero], terminal), axis=0), y=zero)
            ), y=tf.math.greater(x=tf.concat(values=([zero], terminal), axis=0)[-1], y=zero),
            message="Agent.experience: terminal is not the last input timestep."
        ))

        with tf.control_dependencies(control_inputs=assertions):
            # Preprocessing
            for name in self.states_spec:
                if name in self.preprocessing:
                    states[name] = self.preprocessing[name].apply(x=states[name])
            if 'reward' in self.preprocessing:
                reward = self.preprocessing['reward'].apply(x=reward)

            # Core experience
            experienced = self.core_experience(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                terminal=terminal, reward=reward
            )

        # Return
        with tf.control_dependencies(control_inputs=(experienced,)):
            timestep = tf_util.identity(input=self.timesteps)
            episode = tf_util.identity(input=self.episodes)
            update = tf_util.identity(input=self.updates)
        return timestep, episode, update

    @tf_function(num_args=0)
    def update(self):
        # Core update
        updated = self.core_update()

        # Return
        with tf.control_dependencies(control_inputs=(updated,)):
            timestep = tf_util.identity(input=self.timesteps)
            episode = tf_util.identity(input=self.episodes)
            update = tf_util.identity(input=self.updates)
        return timestep, episode, update

    @tf_function(num_args=3)
    def core_act(self, *, states, internals, auxiliaries, deterministic):
        zero = tf_util.constant(value=0, dtype='int')
        past_horizon = tf.math.maximum(
            x=self.policy.past_horizon(on_policy=True),
            y=self.baseline.past_horizon(on_policy=True)
        )
        assertion = tf.debugging.assert_equal(x=past_horizon, y=zero)

        with tf.control_dependencies(control_inputs=(assertion,)):
            batch_size = tf_util.cast(x=tf.shape(input=states.value())[0], dtype='int')
            starts = tf.range(start=batch_size, dtype=tf_util.get_dtype(type='int'))
            lengths = tf_util.ones(shape=(batch_size,), dtype='int')
            horizons = tf.stack(values=(starts, lengths), axis=1)

            # Policy act
            actions, internals['policy'] = self.policy.act(
                states=states, horizons=horizons, internals=internals['policy'],
                auxiliaries=auxiliaries, deterministic=deterministic, return_internals=True
            )

        if self.separate_baseline_policy:
            if len(self.internals_spec['baseline']) > 0:
                # TODO: Baseline policy network apply to retrieve next internals
                _, internals['baseline'] = self.baseline.network.apply(
                    x=states, horizons=horizons, internals=internals['baseline'],
                    return_internals=True
                )
            else:
                internals['baseline'] = TensorDict()

        return actions, internals

    @tf_function(num_args=6)
    def core_observe(self, *, states, internals, auxiliaries, actions, terminal, reward):
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')

        # Experience
        experienced = self.core_experience(
            states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
            terminal=terminal, reward=reward
        )

        # If no periodic update
        if self.update_frequency is None:
            return tf.math.logical_and(x=experienced, y=tf_util.constant(value=False, dtype='bool'))

        # Periodic update
        with tf.control_dependencies(control_inputs=(experienced,)):
            frequency = self.update_frequency.value()
            start = self.update_start.value()

            if self.update_unit == 'timesteps':
                # Timestep-based batch
                past_horizon = tf.math.maximum(
                    x=self.policy.past_horizon(on_policy=False),
                    y=self.baseline.past_horizon(on_policy=False)
                )
                future_horizon = self.estimator.future_horizon()
                start = tf.math.maximum(
                    x=start, y=(frequency + past_horizon + future_horizon + one)
                )
                buffer_observe = tf_util.constant(value=self.config.buffer_observe, dtype='int')
                start = tf.math.maximum(x=start, y=buffer_observe)
                unit = self.timesteps

            elif self.update_unit == 'episodes':
                # Episode-based batch
                start = tf.math.maximum(x=start, y=frequency)
                unit = self.episodes

            unit = unit - start
            is_frequency = tf.math.equal(x=tf.math.mod(x=unit, y=frequency), y=zero)
            is_frequency = tf.math.logical_and(x=is_frequency, y=(unit > self.last_update))

            def perform_update():
                assignment = self.last_update.assign(value=unit, read_value=False)
                with tf.control_dependencies(control_inputs=(assignment,)):
                    return self.core_update()

            def no_update():
                return tf_util.constant(value=False, dtype='bool')

            updated = tf.cond(pred=is_frequency, true_fn=perform_update, false_fn=no_update)

        return updated

    @tf_function(num_args=6)
    def core_experience(self, *, states, internals, auxiliaries, actions, terminal, reward):
        zero = tf_util.constant(value=0, dtype='int')

        # Enqueue experience for early reward estimation
        any_overwritten, overwritten_values = self.estimator.enqueue(
            states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
            terminal=terminal, reward=reward, baseline=self.baseline
        )

        # If terminal, store remaining values in memory

        def true_fn():
            reset_values = self.estimator.terminate(baseline=self.baseline)
            function = (lambda x, y: tf.concat(values=(x, y), axis=0))
            new_overwritten_values = overwritten_values.fmap(
                function=function, zip_values=reset_values
            )
            return new_overwritten_values

        def false_fn():
            return overwritten_values

        with tf.control_dependencies(control_inputs=overwritten_values.flatten()):
            values = tf.cond(
                pred=(tf.concat(values=([zero], terminal), axis=0)[-1] > zero),
                true_fn=true_fn, false_fn=false_fn
            )

        # If any, store overwritten values
        def store():
            return self.memory.enqueue(**values.to_kwargs())

        terminal = values['terminal']
        num_values = tf_util.cast(x=tf.shape(input=terminal)[0], dtype='int')

        stored = tf.cond(pred=(num_values > zero), true_fn=store, false_fn=tf.no_op)

        return stored

    @tf_function(num_args=0)
    def core_update(self):
        one = tf_util.constant(value=1, dtype='int')

        # Retrieve batch
        batch_size = self.update_batch_size.value()
        if self.update_unit == 'timesteps':
            # Timestep-based batch
            # Dependency horizon
            past_horizon = tf.math.maximum(
                x=self.policy.past_horizon(on_policy=False),
                y=self.baseline.past_horizon(on_policy=False)
            )
            future_horizon = self.estimator.future_horizon()
            indices = self.memory.retrieve_timesteps(
                n=batch_size, past_horizon=past_horizon, future_horizon=future_horizon
            )
        elif self.update_unit == 'episodes':
            # Episode-based batch
            indices = self.memory.retrieve_episodes(n=batch_size)

        # Retrieve states and internals
        policy_horizon = self.policy.past_horizon(on_policy=True)
        if self.separate_baseline_policy and self.baseline_optimizer is None:
            assertion = tf.debugging.assert_equal(
                x=policy_horizon, y=self.baseline.past_horizon(on_policy=True),
                message="Policy and baseline depend on a different number of previous states."
            )
            with tf.control_dependencies(control_inputs=(assertion,)):
                policy_horizons, (policy_states,), (policy_internals,) = self.memory.predecessors(
                    indices=indices, horizon=policy_horizon, sequence_values=('states',),
                    initial_values=('internals',)
                )
                baseline_states = policy_states
                baseline_horizons = policy_horizons
                if self.separate_baseline_policy:
                    baseline_internals = policy_internals['baseline']
                else:
                    baseline_internals = policy_internals
        else:
            if self.baseline_optimizer is None:
                policy_horizons, (policy_states,), (policy_internals,) = self.memory.predecessors(
                    indices=indices, horizon=policy_horizon, sequence_values=('states',),
                    initial_values=('internals',)
                )
            else:
                policy_horizons, (policy_states,), (policy_internals,) = self.memory.predecessors(
                    indices=indices, horizon=policy_horizon, sequence_values=('states',),
                    initial_values=('internals/policy',)
                )
            # Optimize !!!!!
            baseline_horizon = self.baseline.past_horizon(on_policy=True)
            if self.separate_baseline_policy:
                baseline_horizons, (baseline_states,), (baseline_internals,) = self.memory.predecessors(
                    indices=indices, horizon=baseline_horizon, sequence_values=('states',),
                    initial_values=('internals/baseline',)
                )
            else:
                baseline_horizons, (baseline_states,), (baseline_internals,) = self.memory.predecessors(
                    indices=indices, horizon=baseline_horizon, sequence_values=('states',),
                    initial_values=('internals/policy',)
                )

        # Retrieve auxiliaries, actions, reward
        auxiliaries, actions, reward = self.memory.retrieve(
            indices=indices, values=('auxiliaries', 'actions', 'reward')
        )

        # Return estimation
        reward = self.estimator.complete_return(
            indices=indices, reward=reward, policy=self.policy, baseline=self.baseline,
            memory=self.memory  # TODO: pass retrieved arguments!
        )
        # reward = self.add_summary(label=('return', 'rewards'), name='return', tensor=reward)  # TODO: need to be moved to episode?
        if 'return' in self.preprocessing:
            # TODO: Can't require state, reset!
            reward = self.preprocessing['return'].apply(x=reward)
            # reward = self.add_summary(
            #     label=('return', 'rewards'), name='preprocessed-return', tensor=reward
            # )

        baseline_arguments = TensorDict(
            states=baseline_states, horizons=baseline_horizons, internals=baseline_internals,
            auxiliaries=auxiliaries, actions=actions, reward=reward
        )
        if self.baseline_objective is not None:
            baseline_arguments['reference'] = self.baseline_objective.reference(
                states=baseline_states, horizons=baseline_horizons, internals=baseline_internals,
                auxiliaries=auxiliaries, actions=actions, reward=reward, policy=self.baseline
            )

        if self.baseline_optimizer is not None:
            def fn_kl_divergence(
                *, states, horizons, internals, auxiliaries, actions, reward, reference
            ):
                reference = self.baseline.kldiv_reference(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
                )
                return self.baseline.kl_divergence(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    reference=reference, reduced=True, return_per_action=False
                )

            if self.baseline_objective is None:
                kwargs = dict()
            else:
                kwargs = self.baseline_objective.optimizer_arguments(policy=self.baseline)
            assert 'source_variables' not in kwargs
            kwargs['source_variables'] = tuple(self.policy.trainable_variables)

            # Optimization
            optimized = self.baseline_optimizer.update(
                arguments=baseline_arguments, variables=tuple(self.baseline.trainable_variables),
                fn_loss=self.baseline_loss, fn_kl_divergence=fn_kl_divergence, **kwargs
            )
            dependencies = (optimized,)
        else:
            dependencies = (reward,)

        with tf.control_dependencies(control_inputs=dependencies):
            if not self.advantage_in_loss:
                # TODO: No separate method? estimator.advantage_in_loss now same function?
                reward = self.estimator.advantage(
                    states=baseline_states, horizons=baseline_horizons,
                    internals=baseline_internals, auxiliaries=auxiliaries, reward=reward,
                    baseline=self.baseline
                )
                # reward = self.add_summary(
                #     label=('advantage', 'rewards'), name='advantage', tensor=reward
                # )
                if 'advantage' in self.preprocessing:
                    # TODO: Can't require state, reset!
                    reward = self.preprocessing['advantage'].apply(x=reward)
                    # reward = self.add_summary(
                    #     label=('advantage', 'rewards'), name='preprocessed-advantage', tensor=reward
                    # )

        if self.baseline_optimizer is None:
            policy_only_internals = policy_internals.get('policy', TensorDict())
        else:
            policy_only_internals = policy_internals
        reference = self.objective.reference(
            states=policy_states, horizons=policy_horizons, internals=policy_only_internals,
            auxiliaries=auxiliaries, actions=actions, reward=reward, policy=self.policy
        )
        if self.separate_baseline_policy and self.baseline_loss_weight is not None and \
                not self.baseline_loss_weight.is_constant(value=0.0):
            reference = TensorDict(policy=reference, baseline=baseline_arguments['reference'])

        policy_arguments = TensorDict(
            states=policy_states, horizons=policy_horizons, internals=policy_internals,
            auxiliaries=auxiliaries, actions=actions, reward=reward, reference=reference
        )

        if self.advantage_in_loss:
            variables = tuple(self.trainable_variables)

            def fn_loss(*, states, horizons, internals, auxiliaries, actions, reward, reference):
                reward = self.estimator.advantage_in_loss(
                    states=states, horizons=horizons, internals=internals['baseline'],
                    auxiliaries=auxiliaries, reward=reward, baseline=self.baseline
                )
                # reward = self.add_summary(
                #     label=('advantage', 'rewards'), name='advantage', tensor=reward
                # )
                if 'advantage' in self.preprocessing:
                    # TODO: Can't require state, reset!
                    reward = self.preprocessing['advantage'].apply(x=reward)
                    # reward = self.add_summary(
                    #     label=('advantage', 'rewards'), name='preprocessed-advantage', tensor=reward
                    # )
                return self.loss(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    actions=actions, reward=reward, reference=reference
                )

        else:
            variables = tuple(self.policy.trainable_variables)
            fn_loss = self.loss

        def fn_kl_divergence(
            *, states, horizons, internals, auxiliaries, actions, reward, reference
        ):
            reference = self.policy.kldiv_reference(
                states=states, horizons=horizons, internals=internals['policy'],
                auxiliaries=auxiliaries
            )
            kl_divergence = self.policy.kl_divergence(
                states=states, horizons=horizons, internals=internals['policy'],
                auxiliaries=auxiliaries, reference=reference, reduced=True, return_per_action=False
            )

            if self.baseline_loss_weight is not None and \
                    not self.baseline_loss_weight.is_constant(value=0.0):
                zero = tf_util.constant(value=0.0, dtype='float')
                baseline_loss_weight = self.baseline_loss_weight.value()

                def no_baseline_kl_divergence():
                    return zero

                def apply_baseline_kl_divergence():
                    reference = self.policy.kldiv_reference(
                        states=states, horizons=horizons, internals=internals['baseline'],
                        auxiliaries=auxiliaries
                    )
                    baseline_kl_divergence = self.baseline.kl_divergence(
                        states=states, horizons=horizons, internals=internals['baseline'],
                        auxiliaries=auxiliaries, reference=reference, reduced=True,
                        return_per_action=False
                    )
                    return baseline_loss_weight * baseline_kl_divergence

                kl_divergence += tf.cond(
                    pred=tf.math.equal(x=baseline_loss_weight, y=zero),
                    true_fn=no_baseline_kl_divergence, false_fn=apply_baseline_kl_divergence
                )
            return kl_divergence

        kwargs = self.objective.optimizer_arguments(policy=self.policy, baseline=self.baseline)
        if self.baseline_objective is not None and self.baseline_loss_weight is not None and \
                not self.baseline_loss_weight.is_constant(value=0.0):
            util.deep_disjoint_update(
                target=kwargs,
                source=self.baseline_objective.optimizer_arguments(policy=self.baseline)
            )

        if self.separate_baseline_policy:
            assert 'source_variables' not in kwargs
            kwargs['source_variables'] = tuple(self.baseline.trainable_variables)
        # if self.global_model is not None:
        #     assert 'global_variables' not in kwargs
        #     kwargs['global_variables'] = tuple(self.global_model.trainable_variables)

        dependencies = policy_arguments.flatten()

        # Variables summaries
        def fn_summary():
            return [
                tf.math.reduce_mean(input_tensor=variable) for variable in self.trainable_variables
            ]

        names = list()
        for variable in self.trainable_variables:
            assert variable.name.startswith(self.name + '/') and variable.name[-2:] == ':0'
            names.append('variables/' + variable.name[len(self.name) + 1: -2] + '-mean')
        dependencies.extend(self.summary(
            label='variables', name=names, data=fn_summary, step='updates', unconditional=True
        ))

        # Hack: KL divergence summary: reference before update
        if self.summary_labels == 'all' or 'kl-divergence' in self.summary_labels:
            kldiv_reference = self.policy.kldiv_reference(
                states=policy_states, horizons=policy_horizons, internals=policy_internals,
                auxiliaries=auxiliaries
            )
            dependencies += kldiv_reference.flatten()

        # Optimization
        with tf.control_dependencies(control_inputs=dependencies):
            optimized = self.optimizer.update(
                arguments=policy_arguments, variables=variables, fn_loss=fn_loss,
                fn_kl_divergence=fn_kl_divergence, **kwargs
            )

        # Update summaries
        with tf.control_dependencies(control_inputs=(optimized,)):

            # KL divergence summaries
            if len(self.actions_spec) > 1:

                def fn_summary():
                    kl_divergence, kl_divergences = self.policy.kl_divergence(
                        states=policy_states, horizons=policy_horizons, internals=policy_internals,
                        auxiliaries=auxiliaries, reference=kldiv_reference, reduced=True,
                        return_per_action=True
                    )
                    kl_divergences = [
                        tf.math.reduce_mean(input_tensor=x) for x in kl_divergences.values()
                    ]
                    kl_divergences.append(tf.math.reduce_mean(input_tensor=kl_divergence))
                    return kl_divergences

                names = ['kl-divergences/' + name for name in self.actions_spec]
                names.append('kl-divergences/overall')

            else:

                def fn_summary():
                    kl_divergence = self.policy.kl_divergence(
                        states=policy_states, horizons=policy_horizons, internals=policy_internals,
                        auxiliaries=auxiliaries, reference=kldiv_reference, reduced=True,
                        return_per_action=False
                    )
                    return tf.math.reduce_mean(input_tensor=kl_divergence)

            self.summary(
                label='kl-divergence', name=names, data=fn_summary, step='updates',
                unconditional=True
            )

        # Increment update
        with tf.control_dependencies(control_inputs=dependencies):
            assignment = self.updates.assign_add(delta=one, read_value=False)

        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf_util.identity(input=optimized)

    @tf_function(num_args=7)
    def loss(self, *, states, horizons, internals, auxiliaries, actions, reward, reference):
        if self.baseline_optimizer is None:
            policy_internals = internals['policy']
        else:
            policy_internals = internals
        if self.separate_baseline_policy and self.baseline_loss_weight is not None and \
                not self.baseline_loss_weight.is_constant(value=0.0):
            policy_reference = reference['policy']
        else:
            policy_reference = reference

        # Loss per instance
        loss = self.objective.loss(
            states=states, horizons=horizons, internals=policy_internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, reference=policy_reference, policy=self.policy
        )

        # Objective loss
        loss = tf.math.reduce_mean(input_tensor=loss, axis=0)
        self.summary(label='loss', name='losses/policy-objective-loss', data=loss, step='updates')

        # Regularization losses
        regularization_loss = self.regularize(
            states=states, horizons=horizons, internals=policy_internals, auxiliaries=auxiliaries
        )
        self.summary(
            label='loss', name='losses/policy-regularization-loss', data=regularization_loss,
            step='updates'
        )
        loss += regularization_loss

        # Baseline loss
        if self.baseline_loss_weight is not None and \
                not self.baseline_loss_weight.is_constant(value=0.0):
            if self.separate_baseline_policy:
                baseline_internals = internals['baseline']
            else:
                baseline_internals = policy_internals
            if self.separate_baseline_policy:
                baseline_reference = reference['baseline']
            else:
                baseline_reference = reference

            zero = tf_util.constant(value=0.0, dtype='float')
            baseline_loss_weight = self.baseline_loss_weight.value()

            def no_baseline_loss():
                return zero

            def apply_baseline_loss():
                baseline_loss = self.baseline_loss(
                    states=states, horizons=horizons, internals=baseline_internals,
                    auxiliaries=auxiliaries, actions=actions, reward=reward,
                    reference=baseline_reference
                )
                return baseline_loss_weight * baseline_loss

            loss += tf.cond(
                pred=tf.math.equal(x=baseline_loss_weight, y=zero),
                true_fn=no_baseline_loss, false_fn=apply_baseline_loss
            )

        self.summary(label='loss', name='losses/policy-loss', data=loss, step='updates')

        return loss

    @tf_function(num_args=4)
    def regularize(self, *, states, horizons, internals, auxiliaries):
        regularization_loss = super().regularize()

        # Entropy regularization
        if not self.entropy_regularization.is_constant(value=0.0):
            zero = tf_util.constant(value=0.0, dtype='float')
            entropy_regularization = self.entropy_regularization.value()

            def no_entropy_regularization():
                return zero

            def apply_entropy_regularization():
                entropy = self.policy.entropy(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    reduced=True, return_per_action=False
                )
                entropy = tf.math.reduce_mean(input_tensor=entropy, axis=0)
                return -entropy_regularization * entropy

            regularization_loss += tf.cond(
                pred=tf.math.equal(x=entropy_regularization, y=zero),
                true_fn=no_entropy_regularization, false_fn=apply_entropy_regularization
            )

        return regularization_loss

    @tf_function(num_args=7)
    def baseline_loss(
        self, *, states, horizons, internals, auxiliaries, actions, reward, reference
    ):
        # Loss per instance
        loss = self.baseline_objective.loss(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, reference=reference, policy=self.baseline
        )

        # Objective loss
        loss = tf.math.reduce_mean(input_tensor=loss)

        if self.separate_baseline_policy:
            self.summary(
                label='loss', name='losses/baseline-objective-loss', data=loss, step='updates'
            )

            # Regularization losses
            regularization_loss = self.baseline.regularize()
            self.summary(
                label='loss', name='losses/baseline-regularization-loss',
                data=regularization_loss, step='updates'
            )
            loss += regularization_loss

        self.summary(label='loss', name='losses/baseline-loss', data=loss, step='updates')

        return loss
