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
from tensorforce.core import Module, parameter_modules


class Estimator(Module):
    """
    Value estimator.

    Args:
        name (string): Estimator name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        values_spec (specification): Values specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        horizon (parameter, long >= 0): Horizon of discounted-sum reward estimation
            (<span style="color:#C00000"><b>required</b></span>).
        discount (parameter, 0.0 <= float <= 1.0): Discount factor for future rewards of
            discounted-sum reward estimation
            (<span style="color:#C00000"><b>required</b></span>).
        estimate_horizon (false | "early" | "late"): Whether to estimate the value of horizon
            states, and if so, whether to estimate early when experience is stored, or late when it
            is retrieved
            (<span style="color:#C00000"><b>required</b></span>).
        estimate_actions (bool): Whether to estimate state-action values instead of state values
            (<span style="color:#C00000"><b>required</b></span>).
        estimate_terminal (bool): Whether to estimate the value of terminal states
            (<span style="color:#C00000"><b>required</b></span>).
        estimate_advantage (bool): Whether to estimate the advantage by subtracting the current
            estimate (<span style="color:#C00000"><b>required</b></span>).
        min_capacity (int > 0): Minimum buffer capacity
            (<span style="color:#0000C0"><b>internal use</b></span>).
        max_past_horizon (int >= 0): Maximum past horizon
            (<span style="color:#0000C0"><b>internal use</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, values_spec, horizon, discount, estimate_horizon, estimate_actions,
        estimate_terminal, estimate_advantage, min_capacity, max_past_horizon, device=None,
        summary_labels=None
    ):
        super().__init__(name=name, device=device, summary_labels=summary_labels)

        self.values_spec = values_spec

        # Horizon
        self.horizon = self.add_module(
            name='horizon', module=horizon, modules=parameter_modules, dtype='long', min_value=0
        )

        # Discount
        self.discount = self.add_module(
            name='discount', module=discount, modules=parameter_modules, dtype='float',
            min_value=0.0, max_value=1.0
        )

        # Baseline settings
        assert estimate_horizon in (False, 'early', 'late')
        self.estimate_horizon = estimate_horizon
        self.estimate_actions = estimate_actions
        self.estimate_terminal = estimate_terminal
        self.estimate_advantage = estimate_advantage

        # Capacity
        if self.estimate_horizon == 'early':
            self.capacity = max(self.horizon.max_value() + 1, min_capacity, max_past_horizon)
        else:
            self.capacity = max(self.horizon.max_value() + 1, min_capacity)

    def min_future_horizon(self):
        if self.estimate_horizon == 'late':
            return self.horizon.min_value() + 1
        else:
            return self.horizon.min_value()

    def max_future_horizon(self):
        if self.estimate_horizon == 'late':
            return self.horizon.max_value() + 1
        else:
            return self.horizon.max_value()

    def tf_future_horizon(self):
        if self.estimate_horizon == 'late':
            return self.horizon.value() + tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        else:
            return self.horizon.value()
    def tf_initialize(self):
        super().tf_initialize()

        # Value buffers
        self.buffers = OrderedDict()
        for name, spec in self.values_spec.items():
            if util.is_nested(name=name):
                self.buffers[name] = OrderedDict()
                for inner_name, spec in spec.items():
                    shape = (self.capacity,) + spec['shape']
                    self.buffers[name][inner_name] = self.add_variable(
                        name=(inner_name + '-buffer'), dtype=spec['type'], shape=shape,
                        is_trainable=False
                    )
            else:
                shape = (self.capacity,) + spec['shape']
                self.buffers[name] = self.add_variable(
                    name=(name + '-buffer'), dtype=spec['type'], shape=shape, is_trainable=False
                )

        # Buffer index (modulo capacity, next index to write to)
        self.buffer_index = self.add_variable(
            name='buffer-index', dtype='long', shape=(), is_trainable=False, initializer='zeros'
        )

    def tf_reset(self, baseline=None):
        # Constants and parameters
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))
        horizon = self.horizon.value()
        discount = self.discount.value()

        # Overwritten buffer indices
        num_overwritten = tf.minimum(x=self.buffer_index, y=capacity)
        indices = tf.range(start=(self.buffer_index - num_overwritten), limit=self.buffer_index)
        indices = tf.math.mod(x=indices, y=capacity)

        # Get overwritten values
        values = OrderedDict()
        for name, buffer in self.buffers.items():
            if util.is_nested(name=name):
                values[name] = OrderedDict()
                for inner_name, buffer in buffer.items():
                    values[name][inner_name] = tf.gather(params=buffer, indices=indices)
            else:
                values[name] = tf.gather(params=buffer, indices=indices)

        states = values['states']
        internals = values['internals']
        auxiliaries = values['auxiliaries']
        actions = values['actions']
        terminal = values['terminal']
        reward = values['reward']
        terminal = values['terminal']

        # Reset buffer index
        with tf.control_dependencies(control_inputs=util.flatten(xs=values)):
            assignment = self.buffer_index.assign(value=zero, read_value=False)

        with tf.control_dependencies(control_inputs=(assignment,)):
            assertions = list()
            # Check whether exactly one terminal (, unless empty?)
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf.math.count_nonzero(input=terminal, dtype=util.tf_dtype(dtype='long')),
                    y=one, message="Timesteps do not contain exactly one terminal."
                )
            )
            # Check whether last value is terminal
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf.math.greater(x=terminal[-1], y=zero),
                    y=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')),
                    message="Terminal is not the last timestep."
                )
            )

        # Get number of values
        with tf.control_dependencies(control_inputs=assertions):
            if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
                num_values = tf.shape(input=terminal, out_type=util.tf_dtype(dtype='long'))[0]
            else:
                num_values = tf.dtypes.cast(
                    x=tf.shape(input=terminal)[0], dtype=util.tf_dtype(dtype='long')
                )

        # Horizon baseline value
        if self.estimate_horizon == 'early' and baseline is not None:
            # Dependency horizon
            # TODO: handle arbitrary non-optimization horizons!
            past_horizon = baseline.past_horizon(is_optimization=False)
            assertion = tf.debugging.assert_equal(
                x=past_horizon, y=zero,
                message="Temporary: baseline cannot depend on previous states."
            )

            # Baseline estimate
            horizon_start = num_values - tf.maximum(x=(num_values - horizon), y=one)
            _states = OrderedDict()
            for name, state in states.items():
                _states[name] = state[horizon_start:]
            _internals = OrderedDict()
            for name, internal in internals.items():
                _internals[name] = internal[horizon_start:]
            _auxiliaries = OrderedDict()
            for name, auxiliary in auxiliaries.items():
                _auxiliaries[name] = auxiliary[horizon_start:]

            with tf.control_dependencies(control_inputs=(assertion,)):
                # some_state = next(iter(states.values()))
                # if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
                #     batch_size = tf.shape(input=some_state, out_type=util.tf_dtype(dtype='long'))[0]
                # else:
                #     batch_size = tf.dtypes.cast(
                #         x=tf.shape(input=some_state)[0], dtype=util.tf_dtype(dtype='long')
                #     )
                batch_size = num_values - horizon_start
                starts = tf.range(start=batch_size, dtype=util.tf_dtype(dtype='long'))
                lengths = tf.ones(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
                Module.update_tensors(dependency_starts=starts, dependency_lengths=lengths)

            if self.estimate_actions:
                _actions = OrderedDict()
                for name, action in actions.items():
                    _actions[name] = action[horizon_start:]
                horizon_estimate = baseline.actions_value(
                    states=_states, internals=_internals, auxiliaries=_auxiliaries, actions=_actions
                )
            else:
                horizon_estimate = baseline.states_value(
                    states=_states, internals=_internals, auxiliaries=_auxiliaries
                )

            # Expand rewards beyond terminal
            terminal_zeros = tf.zeros(shape=(horizon,), dtype=util.tf_dtype(dtype='float'))
            if self.estimate_terminal:
                rewards = tf.concat(
                    values=(reward[:-1], horizon_estimate[-1:], terminal_zeros), axis=0
                )

            else:
                with tf.control_dependencies(control_inputs=(assertion,)):
                    last_reward = tf.where(
                        condition=tf.math.greater(x=terminal[-1], y=one),
                        x=horizon_estimate[-1], y=reward[-1]
                    )
                    rewards = tf.concat(
                        values=(reward[:-1], (last_reward,), terminal_zeros), axis=0
                    )

            # Remove last if necessary
            horizon_end = tf.where(
                condition=tf.math.less_equal(x=num_values, y=horizon), x=zero,
                y=(num_values - horizon)
            )
            horizon_estimate = horizon_estimate[:horizon_end]

            # Expand missing estimates with zeros
            terminal_size = tf.minimum(x=horizon, y=num_values)
            terminal_estimate = tf.zeros(
                shape=(terminal_size,), dtype=util.tf_dtype(dtype='float')
            )
            horizon_estimate = tf.concat(values=(horizon_estimate, terminal_estimate), axis=0)

        else:
            # Expand rewards beyond terminal
            terminal_zeros = tf.zeros(shape=(horizon,), dtype=util.tf_dtype(dtype='float'))
            rewards = tf.concat(values=(reward, terminal_zeros), axis=0)

            # Zero estimate
            horizon_estimate = tf.zeros(shape=(num_values,), dtype=util.tf_dtype(dtype='float'))

        # Calculate discounted sum
        def cond(discounted_sum, horizon):
            return tf.math.greater_equal(x=horizon, y=zero)

        def body(discounted_sum, horizon):
            # discounted_sum = tf.compat.v1.Print(
            #     discounted_sum, (horizon, discounted_sum, rewards[horizon:]), summarize=10
            # )
            discounted_sum = discount * discounted_sum
            discounted_sum = discounted_sum + rewards[horizon: horizon + num_values]
            return discounted_sum, horizon - one

        values['reward'], _ = self.while_loop(
            cond=cond, body=body, loop_vars=(horizon_estimate, horizon), back_prop=False
        )

        return values

    def tf_enqueue(self, states, internals, auxiliaries, actions, terminal, reward, baseline=None):
        # Constants and parameters
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))
        horizon = self.horizon.value()
        discount = self.discount.value()

        assertions = list()
        # Check whether horizon at most capacity
        assertions.append(tf.debugging.assert_less_equal(
            x=horizon, y=capacity,
            message="Estimator capacity has to be at least the same as the estimation horizon."
        ))
        # Check whether at most one terminal
        assertions.append(
            tf.debugging.assert_less_equal(
                x=tf.math.count_nonzero(
                    input=terminal, dtype=util.tf_dtype(dtype='long')
                ), y=one, message="Timesteps contain more than one terminal."
            )
        )
        # Check whether, if any, last value is terminal
        assertions.append(
            tf.debugging.assert_equal(
                x=tf.reduce_any(input_tensor=tf.math.greater(x=terminal, y=zero)),
                y=tf.math.greater(x=terminal[-1], y=zero),
                message="Terminal is not the last timestep."
            )
        )

        # Get number of overwritten values
        with tf.control_dependencies(control_inputs=assertions):
            if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
                num_values = tf.shape(input=terminal, out_type=util.tf_dtype(dtype='long'))[0]
            else:
                num_values = tf.dtypes.cast(
                    x=tf.shape(input=terminal)[0], dtype=util.tf_dtype(dtype='long')
                )
            overwritten_start = tf.maximum(x=self.buffer_index, y=capacity)
            overwritten_limit = tf.maximum(x=(self.buffer_index + num_values), y=capacity)
            num_overwritten = overwritten_limit - overwritten_start

        def update_overwritten_rewards():
            # Get relevant buffer rewards
            buffer_limit = self.buffer_index + tf.minimum(
                x=(num_overwritten + horizon), y=capacity
            )
            buffer_indices = tf.range(start=self.buffer_index, limit=buffer_limit)
            buffer_indices = tf.math.mod(x=buffer_indices, y=capacity)
            rewards = tf.gather(params=self.buffers['reward'], indices=buffer_indices)

            # Get relevant values rewards
            values_limit = tf.maximum(x=(num_overwritten + horizon - capacity), y=zero)
            rewards = tf.concat(values=(rewards, reward[:values_limit]), axis=0)

            # Horizon baseline value
            if self.estimate_horizon == 'early':
                assert baseline is not None
                # Baseline estimate
                buffer_indices = buffer_indices[horizon + one:]
                _states = OrderedDict()
                for name, buffer in self.buffers['states'].items():
                    state = tf.gather(params=buffer, indices=buffer_indices)
                    _states[name] = tf.concat(
                        values=(state, states[name][:values_limit + one]), axis=0
                    )
                _internals = OrderedDict()
                for name, buffer in self.buffers['internals'].items():
                    internal = tf.gather(params=buffer, indices=buffer_indices)
                    _internals[name] = tf.concat(
                        values=(internal, internals[name][:values_limit + one]), axis=0
                    )
                _auxiliaries = OrderedDict()
                for name, buffer in self.buffers['auxiliaries'].items():
                    auxiliary = tf.gather(params=buffer, indices=buffer_indices)
                    _auxiliaries[name] = tf.concat(
                        values=(auxiliary, auxiliaries[name][:values_limit + one]), axis=0
                    )

                # Dependency horizon
                # TODO: handle arbitrary non-optimization horizons!
                past_horizon = baseline.past_horizon(is_optimization=False)
                assertion = tf.debugging.assert_equal(
                    x=past_horizon, y=zero,
                    message="Temporary: baseline cannot depend on previous states."
                )
                with tf.control_dependencies(control_inputs=(assertion,)):
                    some_state = next(iter(_states.values()))
                    if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
                        batch_size = tf.shape(input=some_state, out_type=util.tf_dtype(dtype='long'))[0]
                    else:
                        batch_size = tf.dtypes.cast(
                            x=tf.shape(input=some_state)[0], dtype=util.tf_dtype(dtype='long')
                        )
                    starts = tf.range(start=batch_size, dtype=util.tf_dtype(dtype='long'))
                    lengths = tf.ones(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
                    Module.update_tensors(dependency_starts=starts, dependency_lengths=lengths)

                if self.estimate_actions:
                    _actions = OrderedDict()
                    for name, buffer in self.buffers['actions'].items():
                        action = tf.gather(params=buffer, indices=buffer_indices)
                        _actions[name] = tf.concat(
                            values=(action, actions[name][:values_limit]), axis=0
                        )
                    horizon_estimate = baseline.actions_value(
                        states=_states, internals=_internals, auxiliaries=_auxiliaries,
                        actions=_actions
                    )
                else:
                    horizon_estimate = baseline.states_value(
                        states=_states, internals=_internals, auxiliaries=_auxiliaries
                    )

            else:
                # Zero estimate
                horizon_estimate = tf.zeros(
                    shape=(num_overwritten,), dtype=util.tf_dtype(dtype='float')
                )

            # Calculate discounted sum
            def cond(discounted_sum, horizon):
                return tf.math.greater_equal(x=horizon, y=zero)

            def body(discounted_sum, horizon):
                # discounted_sum = tf.compat.v1.Print(
                #     discounted_sum, (horizon, discounted_sum, rewards[horizon:]), summarize=10
                # )
                discounted_sum = discount * discounted_sum
                discounted_sum = discounted_sum + rewards[horizon: horizon + num_overwritten]
                return discounted_sum, horizon - one

            discounted_sum, _ = self.while_loop(
                cond=cond, body=body, loop_vars=(horizon_estimate, horizon), back_prop=False
            )

            assertions = [
                tf.debugging.assert_equal(
                    x=tf.shape(input=horizon_estimate), y=tf.shape(input=discounted_sum),
                    message="Estimation check."
                ),
                tf.debugging.assert_equal(
                    x=tf.shape(input=rewards, out_type=util.tf_dtype(dtype='long'))[0],
                    y=(horizon + num_overwritten), message="Estimation check."
                )
            ]

            # Overwrite buffer rewards
            with tf.control_dependencies(control_inputs=assertions):
                indices = tf.range(
                    start=self.buffer_index, limit=(self.buffer_index + num_overwritten)
                )
                indices = tf.math.mod(x=indices, y=capacity)
                indices = tf.expand_dims(input=indices, axis=1)

            assignment = self.buffers['reward'].scatter_nd_update(
                indices=indices, updates=discounted_sum
            )

            with tf.control_dependencies(control_inputs=(assignment,)):
                return util.no_operation()

        any_overwritten = tf.math.greater(x=num_overwritten, y=zero)
        updated_rewards = self.cond(
            pred=any_overwritten, true_fn=update_overwritten_rewards, false_fn=util.no_operation
        )

        # Overwritten buffer indices
        with tf.control_dependencies(control_inputs=(updated_rewards,)):
            indices = tf.range(start=overwritten_start, limit=overwritten_limit)
            indices = tf.math.mod(x=indices, y=capacity)

        # Get overwritten values
        with tf.control_dependencies(control_inputs=(indices,)):
            overwritten_values = OrderedDict()
            for name, buffer in self.buffers.items():
                if util.is_nested(name=name):
                    overwritten_values[name] = OrderedDict()
                    for inner_name, buffer in buffer.items():
                        overwritten_values[name][inner_name] = tf.gather(
                            params=buffer, indices=indices
                        )
                else:
                    overwritten_values[name] = tf.gather(params=buffer, indices=indices)

        # Buffer indices to (over)write
        with tf.control_dependencies(control_inputs=util.flatten(xs=overwritten_values)):
            indices = tf.range(start=self.buffer_index, limit=(self.buffer_index + num_values))
            indices = tf.math.mod(x=indices, y=capacity)
            indices = tf.expand_dims(input=indices, axis=1)

        # Write new values
        with tf.control_dependencies(control_inputs=(indices,)):
            values = dict(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                terminal=terminal, reward=reward
            )
            assignments = list()
            for name, buffer in self.buffers.items():
                if util.is_nested(name=name):
                    for inner_name, buffer in buffer.items():
                        assignment = buffer.scatter_nd_update(
                            indices=indices, updates=values[name][inner_name]
                        )
                        assignments.append(assignment)
                else:
                    assignment = buffer.scatter_nd_update(indices=indices, updates=values[name])
                    assignments.append(assignment)

        # Increment buffer index
        with tf.control_dependencies(control_inputs=assignments):
            assignment = self.buffer_index.assign_add(delta=num_values, read_value=False)

        # Return overwritten values or no-op
        with tf.control_dependencies(control_inputs=(assignment,)):
            any_overwritten = tf.math.greater(x=num_overwritten, y=zero)
            overwritten_values = util.fmap(
                function=util.identity_operation, xs=overwritten_values
            )
            return any_overwritten, overwritten_values

    def tf_complete(self, baseline, memory, indices, reward):
        if self.estimate_horizon == 'late':
            assert baseline is not None
            zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
            one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))

            # Baseline dependencies
            past_horizon = baseline.past_horizon(is_optimization=False)
            assertion = tf.debugging.assert_equal(
                x=past_horizon, y=zero,
                message="Temporary: baseline cannot depend on previous states."
            )
            with tf.control_dependencies(control_inputs=(assertion,)):
                if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
                    batch_size = tf.shape(input=reward, out_type=util.tf_dtype(dtype='long'))[0]
                else:
                    batch_size = tf.dtypes.cast(
                        x=tf.shape(input=reward)[0], dtype=util.tf_dtype(dtype='long')
                    )
                starts = tf.range(start=batch_size, dtype=util.tf_dtype(dtype='long'))
                lengths = tf.ones(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
                Module.update_tensors(dependency_starts=starts, dependency_lengths=lengths)

            horizon = self.horizon.value()
            discount = self.discount.value()

            if self.estimate_actions:
                # horizon change: see timestep-based batch sampling
                horizons, (states, internals, auxiliaries, terminal) = memory.successors(
                    indices=indices, horizon=(horizon + one),
                    final_values=('states', 'internals', 'auxiliaries', 'terminal')
                )
                # TODO: Double DQN would require main policy here
                actions = baseline.act(
                    states=states, internals=internals, auxiliaries=auxiliaries,
                    return_internals=False
                )
                horizon_estimate = baseline.actions_value(
                    states=states, internals=internals, auxiliaries=auxiliaries, actions=actions
                )
            else:
                # horizon change: see timestep-based batch sampling
                horizons, (states, internals, auxiliaries, terminal) = memory.successors(
                    indices=indices, horizon=(horizon + one),
                    final_values=('states', 'internals', 'auxiliaries', 'terminal')
                )
                horizon_estimate = baseline.states_value(
                    states=states, internals=internals, auxiliaries=auxiliaries
                )

            exponent = tf.dtypes.cast(x=horizons, dtype=util.tf_dtype(dtype='float'))
            discounts = tf.math.pow(x=discount, y=exponent)
            if not self.estimate_terminal:
                with tf.control_dependencies(control_inputs=(assertion,)):
                    discounts = tf.where(
                        condition=tf.math.equal(x=terminal, y=one),
                        x=tf.zeros_like(input=discounts, dtype=util.tf_dtype(dtype='float')),
                        y=discounts
                    )
            reward = reward + discounts * tf.stop_gradient(input=horizon_estimate)
            # TODO: stop gradients?

        return reward

    def tf_estimate(self, baseline, memory, indices, reward, is_baseline_optimized):
        if self.estimate_advantage:
            assert baseline is not None
            zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))

            # Baseline dependencies
            past_horizon = baseline.past_horizon(is_optimization=is_baseline_optimized)
            assertion = tf.debugging.assert_equal(
                x=past_horizon, y=zero,
                message="Temporary: baseline cannot depend on previous states."
            )
            with tf.control_dependencies(control_inputs=(assertion,)):
                if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
                    batch_size = tf.shape(input=reward, out_type=util.tf_dtype(dtype='long'))[0]
                else:
                    batch_size = tf.dtypes.cast(
                        x=tf.shape(input=reward)[0], dtype=util.tf_dtype(dtype='long')
                    )
                starts = tf.range(start=batch_size, dtype=util.tf_dtype(dtype='long'))
                lengths = tf.ones(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
                Module.update_tensors(dependency_starts=starts, dependency_lengths=lengths)

            if self.estimate_actions:
                states, internals, auxiliaries, actions = memory.retrieve(
                    indices=indices, values=('states', 'internals', 'auxiliaries', 'actions')
                )
                critic_estimate = baseline.actions_value(
                    states=states, internals=internals, auxiliaries=auxiliaries, actions=actions
                )
            else:
                states, internals, auxiliaries = memory.retrieve(
                    indices=indices, values=('states', 'internals', 'auxiliaries')
                )
                critic_estimate = baseline.states_value(
                    states=states, internals=internals, auxiliaries=auxiliaries
                )

            reward = reward - critic_estimate

        return reward
