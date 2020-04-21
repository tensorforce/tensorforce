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

import tensorflow as tf

from tensorforce import util
from tensorforce.core import Module, parameter_modules, SignatureDict, TensorDict, TensorSpec, \
    tf_function, tf_util, VariableDict


class Estimator(Module):
    """
    Value estimator.

    Args:
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
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        values_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        min_capacity (int > 0): <span style="color:#0000C0"><b>internal use</b></span>.
        max_past_horizon (int >= 0): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, horizon, discount, estimate_horizon, estimate_actions, estimate_terminal,
        estimate_advantage, device=None, summary_labels=None, name=None, values_spec=None,
        min_capacity=None, max_past_horizon=None
    ):
        super().__init__(device=device, summary_labels=summary_labels, name=name)

        self.values_spec = values_spec

        # Horizon
        self.horizon = self.add_module(
            name='horizon', module=horizon, modules=parameter_modules, dtype='int', min_value=0
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
            # max_past_horizon: baseline on-policy horizon
            self.capacity = max(self.horizon.max_value() + 1, min_capacity, max_past_horizon)
        else:
            self.capacity = max(self.horizon.max_value() + 1, min_capacity)

    def max_future_horizon(self):
        if self.estimate_horizon == 'late':
            return self.horizon.max_value() + 1
        else:
            return 0

    def initialize(self):
        super().initialize()

        # Value buffers
        function = (lambda name, spec: self.variable(
            name=(name.replace('/', '_') + '-buffer'), dtype=spec.type,
            shape=((self.capacity,) + spec.shape), initializer='zeros', is_trainable=False,
            is_saved=False
        ))
        self.buffers = self.values_spec.fmap(function=function, cls=VariableDict, with_names=True)

        # Buffer index (modulo capacity, next index to write to)
        self.buffer_index = self.variable(
            name='buffer-index', dtype='int', shape=(), initializer='zeros', is_trainable=False,
            is_saved=False
        )

    def input_signature(self, *, function):
        if function == 'advantage':
            return SignatureDict(
                indices=TensorSpec(type='int', shape=()).signature(batched=True),
                reward=self.values_spec['reward'].signature(batched=True)
            )

        elif function == 'complete_return':
            return SignatureDict(
                indices=TensorSpec(type='int', shape=()).signature(batched=True),
                reward=self.values_spec['reward'].signature(batched=True)
            )

        elif function == 'enqueue':
            return self.values_spec.signature(batched=True)

        elif function == 'future_horizon':
            return SignatureDict()

        elif function == 'reset':
            return SignatureDict()

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=0)
    def future_horizon(self):
        if self.estimate_horizon == 'late':
            return self.horizon.value() + tf_util.constant(value=1, dtype='int')
        else:
            return tf_util.constant(value=0, dtype='int')

    @tf_function(num_args=0)
    def reset(self, *, baseline):
        # Constants and parameters
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        capacity = tf_util.constant(value=self.capacity, dtype='int')
        horizon = self.horizon.value()
        discount = self.discount.value()

        # Overwritten buffer indices
        num_overwritten = tf.math.minimum(x=self.buffer_index, y=capacity)
        indices = tf.range(start=(self.buffer_index - num_overwritten), limit=self.buffer_index)
        indices = tf.math.mod(x=indices, y=capacity)

        # Get overwritten values
        values = self.buffers.fmap(
            function=(lambda buffer: tf.gather(params=buffer, indices=indices)), cls=TensorDict
        )

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
                    x=tf.math.count_nonzero(input=terminal, dtype=tf_util.get_dtype(type='int')),
                    y=one, message="Timesteps do not contain exactly one terminal."
                )
            )
            # Check whether last value is terminal
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf.math.greater(x=terminal[-1], y=zero),
                    y=tf_util.constant(value=True, dtype='bool'),
                    message="Terminal is not the last timestep."
                )
            )

        # Get number of values
        with tf.control_dependencies(control_inputs=assertions):
            num_values = tf_util.cast(x=tf.shape(input=terminal)[0], dtype='int')

        # Horizon baseline value
        if self.estimate_horizon == 'early' and baseline is not None:
            # Dependency horizon
            # TODO: handle arbitrary non-optimization horizons!
            past_horizon = baseline.past_horizon(on_policy=True)
            assertion = tf.debugging.assert_equal(
                x=past_horizon, y=zero,
                message="Temporary: baseline cannot depend on previous states."
            )

            # Baseline estimate
            horizon_start = num_values - tf.math.maximum(x=(num_values - horizon), y=one)
            function = (lambda value: value[horizon_start:])
            _states = TensorDict()
            _states = states.fmap(function=function)
            _internals = internals['baseline'].fmap(function=function)
            _auxiliaries = auxiliaries.fmap(function=function)

            with tf.control_dependencies(control_inputs=(assertion,)):
                batch_size = num_values - horizon_start
                starts = tf.range(start=batch_size)
                lengths = tf_util.ones(shape=(batch_size,), dtype='int')
                horizons = tf.stack(values=(starts, lengths), axis=1)

            if self.estimate_actions:
                _actions = actions.fmap(function=function)
                horizon_estimate = baseline.actions_value(
                    states=_states, horizons=horizons, internals=_internals,
                    auxiliaries=_auxiliaries, actions=_actions, reduced=True,
                    return_per_action=False
                )
            else:
                horizon_estimate = baseline.states_value(
                    states=_states, horizons=horizons, internals=_internals,
                    auxiliaries=_auxiliaries, reduced=True, return_per_action=False
                )

            # Expand rewards beyond terminal
            terminal_zeros = tf_util.zeros(shape=(horizon,), dtype='float')
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
            terminal_size = tf.math.minimum(x=horizon, y=num_values)
            terminal_estimate = tf_util.zeros(shape=(terminal_size,), dtype='float')
            horizon_estimate = tf.concat(values=(horizon_estimate, terminal_estimate), axis=0)

        else:
            # Expand rewards beyond terminal
            terminal_zeros = tf_util.zeros(shape=(horizon,), dtype='float')
            rewards = tf.concat(values=(reward, terminal_zeros), axis=0)

            # Zero estimate
            horizon_estimate = tf_util.zeros(shape=(num_values,), dtype='float')

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

        values['reward'], _ = tf.while_loop(
            cond=cond, body=body, loop_vars=(horizon_estimate, horizon), back_prop=False
        )

        return values

    @tf_function(num_args=6)
    def enqueue(self, *, states, internals, auxiliaries, actions, terminal, reward, baseline):
        # Constants and parameters
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        capacity = tf_util.constant(value=self.capacity, dtype='int')
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
                x=tf.math.count_nonzero(input=terminal, dtype=tf_util.get_dtype(type='int')),
                y=one, message="Timesteps contain more than one terminal."
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
            num_values = tf_util.cast(x=tf.shape(input=terminal)[0], dtype='int')
            overwritten_start = tf.math.maximum(x=self.buffer_index, y=capacity)
            overwritten_limit = tf.math.maximum(x=(self.buffer_index + num_values), y=capacity)
            num_overwritten = overwritten_limit - overwritten_start

        def update_overwritten_rewards():
            # Get relevant buffer rewards
            buffer_limit = self.buffer_index + tf.math.minimum(
                x=(num_overwritten + horizon), y=capacity
            )
            buffer_indices = tf.range(start=self.buffer_index, limit=buffer_limit)
            buffer_indices = tf.math.mod(x=buffer_indices, y=capacity)
            rewards = tf.gather(params=self.buffers['reward'], indices=buffer_indices)

            # Get relevant values rewards
            values_limit = tf.math.maximum(x=(num_overwritten + horizon - capacity), y=zero)
            rewards = tf.concat(values=(rewards, reward[:values_limit]), axis=0)

            # Horizon baseline value
            if self.estimate_horizon == 'early':
                assert baseline is not None
                # Baseline estimate
                buffer_indices = buffer_indices[horizon + one:]
                function = (lambda value, buffer: tf.concat(values=(
                    tf.gather(params=buffer, indices=buffer_indices), value[:values_limit + one]
                ), axis=0))
                _states = states.fmap(function=function, zip_values=self.buffers['states'])
                _internals = internals['baseline'].fmap(
                    function=function, zip_values=self.buffers['internals/baseline']
                )
                _auxiliaries = auxiliaries.fmap(
                    function=function, zip_values=self.buffers['auxiliaries']
                )

                # Dependency horizon
                # TODO: handle arbitrary non-optimization horizons!
                past_horizon = baseline.past_horizon(on_policy=True)
                assertion = tf.debugging.assert_equal(
                    x=past_horizon, y=zero,
                    message="Temporary: baseline cannot depend on previous states."
                )
                with tf.control_dependencies(control_inputs=(assertion,)):
                    batch_size = tf_util.cast(x=tf.shape(input=_states.value())[0], dtype='int')
                    starts = tf.range(start=batch_size)
                    lengths = tf_util.ones(shape=(batch_size,), dtype='int')
                    horizons = tf.stack(values=(starts, lengths), axis=1)
                    # print('!!!', horizons, horizons.graph)

                if self.estimate_actions:
                    _actions = actions.fmap(function=function, zip_values=self.buffers['actions'])
                    horizon_estimate = baseline.actions_value(
                        states=_states, horizons=horizons, internals=_internals,
                        auxiliaries=_auxiliaries, actions=_actions, reduced=True,
                        return_per_action=False
                    )
                else:
                    horizon_estimate = baseline.states_value(
                        states=_states, horizons=horizons, internals=_internals,
                        auxiliaries=_auxiliaries, reduced=True, return_per_action=False
                    )

            else:
                # Zero estimate
                horizon_estimate = tf_util.zeros(shape=(num_overwritten,), dtype='float')

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

            discounted_sum, _ = tf.while_loop(
                cond=cond, body=body, loop_vars=(horizon_estimate, horizon), back_prop=False
            )

            assertions = [
                tf.debugging.assert_equal(
                    x=tf.shape(input=horizon_estimate), y=tf.shape(input=discounted_sum),
                    message="Estimation check."
                ),
                tf.debugging.assert_equal(
                    x=tf_util.cast(x=tf.shape(input=rewards)[0], dtype='int'),
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
                return tf.no_op()

        any_overwritten = tf.math.greater(x=num_overwritten, y=zero)
        updated_rewards = tf.cond(
            pred=any_overwritten, true_fn=update_overwritten_rewards, false_fn=tf.no_op
        )

        # Overwritten buffer indices
        with tf.control_dependencies(control_inputs=(updated_rewards,)):
            indices = tf.range(start=overwritten_start, limit=overwritten_limit)
            indices = tf.math.mod(x=indices, y=capacity)

        # Get overwritten values
        with tf.control_dependencies(control_inputs=(indices,)):
            function = (lambda buffer: tf.gather(params=buffer, indices=indices))
            overwritten_values = self.buffers.fmap(function=function, cls=TensorDict)

        # Buffer indices to (over)write
        with tf.control_dependencies(control_inputs=util.flatten(xs=overwritten_values)):
            indices = tf.range(start=self.buffer_index, limit=(self.buffer_index + num_values))
            indices = tf.math.mod(x=indices, y=capacity)
            indices = tf.expand_dims(input=indices, axis=1)

        # Write new values
        with tf.control_dependencies(control_inputs=(indices,)):
            values = TensorDict(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                terminal=terminal, reward=reward
            )
            function = (lambda value, buffer: buffer.scatter_nd_update(
                indices=indices, updates=value
            ))
            assignments = values.fmap(function=function, cls=list, zip_values=self.buffers)

        # Increment buffer index
        with tf.control_dependencies(control_inputs=assignments):
            assignment = self.buffer_index.assign_add(delta=num_values, read_value=False)

        # Return overwritten values or no-op
        with tf.control_dependencies(control_inputs=(assignment,)):
            any_overwritten = tf.math.greater(x=num_overwritten, y=zero)
            overwritten_values = overwritten_values.fmap(function=tf_util.identity)
            return any_overwritten, overwritten_values

    @tf_function(num_args=2)
    def complete_return(self, *, indices, reward, baseline, memory):
        if self.estimate_horizon == 'late':
            assert baseline is not None
            zero = tf_util.constant(value=0, dtype='int')
            one = tf_util.constant(value=1, dtype='int')

            # Baseline dependencies
            # TODO: handle arbitrary non-optimization horizons! or re-compute internals, hence off?
            past_horizon = baseline.past_horizon(on_policy=True)
            assertion = tf.debugging.assert_equal(
                x=past_horizon, y=zero,
                message="Temporary: baseline cannot depend on previous states."
            )
            with tf.control_dependencies(control_inputs=(assertion,)):
                batch_size = tf_util.cast(x=tf.shape(input=reward)[0], dtype='int')
                starts = tf.range(start=batch_size)
                lengths = tf_util.ones(shape=(batch_size,), dtype='int')
                horizons = tf.stack(values=(starts, lengths), axis=1)

            horizon = self.horizon.value()
            discount = self.discount.value()

            if self.estimate_actions:
                # horizon change: see timestep-based batch sampling
                final_horizons, (states, internals, auxiliaries, terminal) = memory.successors(
                    indices=indices, horizon=(horizon + one), sequence_values=(),
                    final_values=('states', 'internals/baseline', 'auxiliaries', 'terminal')
                )
                # TODO: Double DQN would require main policy here
                actions = baseline.act(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    deterministic=True, return_internals=False
                )
                horizon_estimate = baseline.actions_value(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    actions=actions, reduced=True, return_per_action=False
                )
            else:
                # horizon change: see timestep-based batch sampling
                final_horizons, (states, internals, auxiliaries, terminal) = memory.successors(
                    indices=indices, horizon=(horizon + one), sequence_values=(),
                    final_values=('states', 'baseline/internals', 'auxiliaries', 'terminal')
                )
                horizon_estimate = baseline.states_value(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    reduced=True, return_per_action=False
                )

            exponent = tf_util.cast(x=final_horizons, dtype='float')
            discounts = tf.math.pow(x=discount, y=exponent)
            if not self.estimate_terminal:
                with tf.control_dependencies(control_inputs=(assertion,)):
                    discounts = tf.where(
                        condition=tf.math.equal(x=terminal, y=one),
                        x=tf.zeros_like(input=discounts), y=discounts
                    )
            reward = reward + discounts * horizon_estimate
            # TODO: stop gradients?

        return reward

    @tf_function(num_args=2)
    def advantage(self, *, indices, reward, baseline, memory, is_baseline_optimized):
        if self.estimate_advantage:
            assert baseline is not None
            # zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='int'))

            # with tf.control_dependencies(control_inputs=(assertion,)):
            # if util.tf_dtype(dtype='int') in (tf.int32, tf.int64):
            #     batch_size = tf.shape(input=reward, out_type=util.tf_dtype(dtype='int'))[0]
            # else:
            #     batch_size = tf.dtypes.cast(
            #         x=tf.shape(input=reward)[0], dtype=util.tf_dtype(dtype='int')
            #     )

            # Baseline dependencies
            # TODO: or re-compute internals, hence always off-policy?
            past_horizon = baseline.past_horizon(on_policy=(not is_baseline_optimized))
            horizons, (states,), (internals,) = memory.predecessors(
                indices=indices, horizon=past_horizon, sequence_values=('states',),
                initial_values=('internals/baseline',)
            )

            if self.estimate_actions:
                auxiliaries, actions = memory.retrieve(
                    indices=indices, values=('auxiliaries', 'actions')
                )
                critic_estimate = baseline.actions_value(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    actions=actions, reduced=True, return_per_action=False
                )
            else:
                (auxiliaries,) = memory.retrieve(indices=indices, values=('auxiliaries',))
                critic_estimate = baseline.states_value(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    reduced=True, return_per_action=False
                )

            reward = reward - critic_estimate

        return reward
