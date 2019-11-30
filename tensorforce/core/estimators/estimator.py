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
from tensorforce.core.utils import CircularBuffer


class Estimator(CircularBuffer):
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
            estimate
            (<span style="color:#C00000"><b>required</b></span>).
        capacity (int > 0): Estimation buffer capacity
            (<span style="color:#0000C0"><b>internal use</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, values_spec, horizon, discount, estimate_horizon, estimate_actions,
        estimate_terminal, estimate_advantage, capacity=None, device=None, summary_labels=None
    ):
        if capacity is None:
            if not isinstance(horizon, int):
                raise TensorforceError.unexpected()
            capacity = horizon

        super().__init__(
            name=name, capacity=capacity, values_spec=values_spec, return_overwritten=True,
            device=device, summary_labels=summary_labels
        )

        # Horizon
        self.horizon = self.add_module(
            name='horizon', module=horizon, modules=parameter_modules, dtype='long'
        )

        # Discount
        self.discount = self.add_module(
            name='discount', module=discount, modules=parameter_modules, dtype='float'
        )

        # Baseline settings
        assert estimate_horizon in (False, 'early', 'late')
        self.estimate_horizon = estimate_horizon
        self.estimate_actions = estimate_actions
        self.estimate_terminal = estimate_terminal
        self.estimate_advantage = estimate_advantage

    def tf_reset(self, baseline=None):
        values = super().tf_reset()

        # Constants and parameters
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        horizon = self.horizon.value()
        discount = self.discount.value()

        assertions = list()
        # Check whether exactly one terminal, unless empty
        assertions.append(
            tf.debugging.assert_equal(
                x=tf.math.count_nonzero(
                    input=values['terminal'], dtype=util.tf_dtype(dtype='long')
                ), y=one
            )
        )
        # Check whether last value is terminal
        assertions.append(
            tf.debugging.assert_equal(
                x=tf.math.greater(x=values['terminal'][-1], y=zero),
                y=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
            )
        )

        # Get number of values
        with tf.control_dependencies(control_inputs=assertions):
            value = values['terminal']
            if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
                num_values = tf.shape(input=value, out_type=util.tf_dtype(dtype='long'))[0]
            else:
                num_values = tf.dtypes.cast(
                    x=tf.shape(input=value)[0], dtype=util.tf_dtype(dtype='long')
                )

        # Horizon baseline value
        if self.estimate_horizon == 'early' and baseline is not None:
            # Baseline estimate
            horizon_start = num_values - tf.maximum(x=(num_values - horizon), y=one)
            states = OrderedDict()
            for name, state in values['states'].items():
                states[name] = state[horizon_start:]
            internals = OrderedDict()
            for name, internal in values['internals'].items():
                internals[name] = internal[horizon_start:]
            auxiliaries = OrderedDict()
            for name, auxiliary in values['auxiliaries'].items():
                auxiliaries[name] = auxiliary[horizon_start:]

            # Dependency horizon
            # TODO: handle arbitrary non-optimization horizons!
            dependency_horizon = baseline.dependency_horizon(is_optimization=False)
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

            if self.estimate_actions:
                actions = OrderedDict()
                for name, action in values['actions'].items():
                    actions[name] = action[horizon_start:]
                horizon_estimate = baseline.actions_value(
                    states=states, internals=internals, auxiliaries=auxiliaries, actions=actions
                )
            else:
                horizon_estimate = baseline.states_value(
                    states=states, internals=internals, auxiliaries=auxiliaries
                )


            # terminal_size = tf.minimum(x=horizon, y=num_values)
            # if self.estimate_terminal:
            #     terminal_estimate = horizon_estimate[-1]
            #     terminal_estimate = tf.fill(dims=(terminal_size,), value=terminal_estimate)
            #     horizon_end = tf.where(condition=(num_values <= horizon), x=zero, y=(num_values - horizon))
            #     horizon_estimate = horizon_estimate[:horizon_end]
            # else:
            #     terminal_estimate = tf.zeros(
            #         shape=(terminal_size,), dtype=util.tf_dtype(dtype='float')
            #     )
            # horizon_estimate = tf.concat(values=(horizon_estimate, terminal_estimate), axis=0)


            # Expand rewards beyond terminal
            terminal_zeros = tf.zeros(shape=(horizon,), dtype=util.tf_dtype(dtype='float'))
            if self.estimate_terminal:
                rewards = tf.concat(
                    values=(values['reward'][:-1], horizon_estimate[-1:], terminal_zeros), axis=0
                )

            else:
                with tf.control_dependencies(control_inputs=(assertion,)):
                    last_reward = tf.where(
                        condition=tf.math.greater(x=values['terminal'][-1], y=one),
                        x=horizon_estimate[-1], y=values['reward'][-1]
                    )
                    rewards = tf.concat(
                        values=(values['reward'][:-1], (last_reward,), terminal_zeros), axis=0
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
            rewards = tf.concat(values=(values['reward'], terminal_zeros), axis=0)

            # Zero estimate
            horizon_estimate = tf.zeros(shape=(num_values,), dtype=util.tf_dtype(dtype='float'))

        # Calculate discounted sum
        def cond(discounted_sum, horizon):
            return tf.math.greater_equal(x=horizon, y=zero)

        def body(discounted_sum, horizon):
            # discounted_sum = tf.Print(discounted_sum, (horizon, discounted_sum, rewards[horizon:]), summarize=10)
            discounted_sum = discount * discounted_sum
            discounted_sum = discounted_sum + rewards[horizon: horizon + num_values]
            return discounted_sum, horizon - one

        values['reward'], _ = self.while_loop(
            cond=cond, body=body, loop_vars=(horizon_estimate, horizon), back_prop=False
        )

        return values

    def tf_enqueue(self, baseline=None, **values):
        # Constants and parameters
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))
        horizon = self.horizon.value()
        discount = self.discount.value()

        assertions = list()
        # Check whether horizon at most capacity
        assertions.append(tf.debugging.assert_less_equal(x=horizon, y=capacity))
        # Check whether at most one terminal
        assertions.append(
            tf.debugging.assert_less_equal(
                x=tf.math.count_nonzero(
                    input=values['terminal'], dtype=util.tf_dtype(dtype='long')
                ), y=one
            )
        )
        # Check whether, if any, last value is terminal
        assertions.append(
            tf.debugging.assert_equal(
                x=tf.reduce_any(input_tensor=tf.math.greater(x=values['terminal'], y=zero)),
                y=tf.math.greater(x=values['terminal'][-1], y=zero)
            )
        )

        # Get number of overwritten values
        with tf.control_dependencies(control_inputs=assertions):
            value = values['terminal']
            if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
                num_values = tf.shape(input=value, out_type=util.tf_dtype(dtype='long'))[0]
            else:
                num_values = tf.dtypes.cast(
                    x=tf.shape(input=value)[0], dtype=util.tf_dtype(dtype='long')
                )
            start = tf.maximum(x=self.buffer_index, y=capacity)
            limit = tf.maximum(x=(self.buffer_index + num_values), y=capacity)
            num_overwritten = limit - start

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
            rewards = tf.concat(values=(rewards, values['reward'][:values_limit]), axis=0)

            # Horizon baseline value
            if self.estimate_horizon == 'early':
                assert baseline is not None
                # Baseline estimate
                buffer_indices = buffer_indices[horizon + one:]
                states = OrderedDict()
                for name, buffer in self.buffers['states'].items():
                    state = tf.gather(params=buffer, indices=buffer_indices)
                    states[name] = tf.concat(
                        values=(state, values['states'][name][:values_limit]), axis=0
                    )
                internals = OrderedDict()
                for name, buffer in self.buffers['internals'].items():
                    internal = tf.gather(params=buffer, indices=buffer_indices)
                    internals[name] = tf.concat(
                        values=(internal, values['internals'][name][:values_limit]), axis=0
                    )
                auxiliaries = OrderedDict()
                for name, buffer in self.buffers['auxiliaries'].items():
                    auxiliary = tf.gather(params=buffer, indices=buffer_indices)
                    auxiliaries[name] = tf.concat(
                        values=(auxiliary, values['auxiliaries'][name][:values_limit]), axis=0
                    )

                # Dependency horizon
                # TODO: handle arbitrary non-optimization horizons!
                dependency_horizon = baseline.dependency_horizon(is_optimization=False)
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

                if self.estimate_actions:
                    actions = OrderedDict()
                    for name, buffer in self.buffers['actions'].items():
                        action = tf.gather(params=buffer, indices=buffer_indices)
                        actions[name] = tf.concat(
                            values=(action, values['actions'][name][:values_limit]), axis=0
                        )
                    horizon_estimate = baseline.actions_value(
                        states=states, internals=internals, auxiliaries=auxiliaries,
                        actions=actions
                    )
                else:
                    horizon_estimate = baseline.states_value(
                        states=states, internals=internals, auxiliaries=auxiliaries
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
                # discounted_sum = tf.Print(discounted_sum, (horizon, discounted_sum, rewards[horizon:]), summarize=10)
                discounted_sum = discount * discounted_sum
                discounted_sum = discounted_sum + rewards[horizon: horizon + num_overwritten]
                return discounted_sum, horizon - one

            discounted_sum, _ = self.while_loop(
                cond=cond, body=body, loop_vars=(horizon_estimate, horizon), back_prop=False
            )

            assertions = list()
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf.shape(input=horizon_estimate), y=tf.shape(input=discounted_sum)
                )
            )
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf.shape(input=rewards, out_type=util.tf_dtype(dtype='long'))[0],
                    y=(horizon + num_overwritten)
                )
            )

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

        with tf.control_dependencies(control_inputs=(updated_rewards,)):
            return super().tf_enqueue(**values)

    def tf_complete(self, baseline, memory, indices, reward):
        if self.estimate_horizon == 'late':
            assert baseline is not None
            zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
            one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))

            # Baseline dependencies
            dependency_horizon = baseline.dependency_horizon(is_optimization=False)
            assertion = tf.debugging.assert_equal(x=dependency_horizon, y=zero)
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

    def tf_estimate(self, baseline, memory, indices, reward):
        if self.estimate_advantage:
            assert baseline is not None
            zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))

            # Baseline dependencies
            dependency_horizon = baseline.dependency_horizon(is_optimization=False)
            assertion = tf.debugging.assert_equal(x=dependency_horizon, y=zero)
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
