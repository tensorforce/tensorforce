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

from tensorforce import TensorforceError, util
from tensorforce.core import memory_modules, Module, optimizer_modules, parameter_modules
from tensorforce.core.models import Model


class MemoryModel(Model):
    """
    A memory model is a generic model to accumulate and sample data.

    Child classes need to implement the following methods:
    - `tf_loss_per_instance(states, internals, actions, terminal, reward)` returning the loss
        per instance for a batch.
    - `tf_regularization_losses(states, internals)` returning a dict of regularization losses.
    """

    def __init__(
        self,
        # Model
        states, actions, scope, device, saver, summarizer, execution, parallel_interactions,
        buffer_observe, exploration, variable_noise, states_preprocessing, reward_preprocessing,
        # MemoryModel
        update_mode, memory, optimizer, discount
    ):
        """
        Memory model.

        Args:
            states (spec): The state-space description dictionary.
            actions (spec): The action-space description dictionary.
            scope (str): The root scope str to use for tf variable scoping.
            device (str): The name of the device to run the graph of this model on.
            saver (spec): Dict specifying whether and how to save the model's parameters.
            summarizer (spec): Dict specifying which tensorboard summaries should be created and added to the graph.
            execution (spec): Dict specifying whether and how to do distributed training on the model's graph.
            batching_capacity (int): Batching capacity.
            variable_noise (float): The stddev value of a Normal distribution used for adding random
                noise to the model's output (for each batch, noise can be toggled and - if active - will be resampled).
                Use None for not adding any noise.
            states_preprocessing (spec / dict of specs): Dict specifying whether and how to preprocess state signals
                (e.g. normalization, greyscale, etc..).
            actions_exploration (spec / dict of specs): Dict specifying whether and how to add exploration to the model's
                "action outputs" (e.g. epsilon-greedy).
            reward_preprocessing (spec): Dict specifying whether and how to preprocess rewards coming
                from the Environment (e.g. reward normalization).
            update_mode (spec): Update mode.
            memory (spec): Memory.
            optimizer (spec): Dict specifying the tf optimizer to use for tuning the model's trainable parameters.
            discount (float): The RL reward discount factor (gamma).
        """
        super().__init__(
            # Model
            states=states, actions=actions, scope=scope, device=device, saver=saver,
            summarizer=summarizer, execution=execution,
            parallel_interactions=parallel_interactions, buffer_observe=buffer_observe,
            exploration=exploration, variable_noise=variable_noise,
            states_preprocessing=states_preprocessing, reward_preprocessing=reward_preprocessing
        )

        # Update mode
        if not all(
            key in ('unit', 'sequence_length', 'batch_size', 'frequency', 'start')
            for key in update_mode
        ):
            raise TensorforceError.value(name='update_mode', value=list(update_mode))
        # update_mode: unit
        elif 'unit' not in update_mode:
            raise TensorforceError.required(name='update_mode', value='unit')
        elif update_mode['unit'] not in ('timesteps', 'episodes', 'sequences'):
            raise TensorforceError.value(
                name='update_mode', argument='unit', value=update_mode['unit']
            )
        # update_mode: sequence_length
        elif update_mode['unit'] == 'sequences' and 'sequence_length' not in update_mode:
            raise TensorforceError.required(name='update_mode', value='sequence_length')
        elif update_mode['unit'] != 'sequences' and 'sequence_length' in update_mode:
            raise TensorforceError.value(name='update_mode', value='sequence_length')
        # elif 'sequence_length' in update_mode and \
        #         (not isinstance(update_mode['sequence_length'], int) or update_mode['sequence_length'] <= 1):
        #     raise TensorforceError.value(
        #         name='update_mode', argument='sequence_length',
        #         value=update_mode['sequence_length']
        #     )
        # update_mode: batch_size
        elif 'batch_size' not in update_mode:
            raise TensorforceError.required(name='update_mode', value='batch_size')
        # elif not isinstance(update_mode['batch_size'], int) or update_mode['batch_size'] < 1:
        #     raise TensorforceError.value(
        #         name='update_mode', argument='batch_size', value=update_mode['batch_size']
        #     )
        # update_mode: frequency
        # elif 'frequency' in update_mode and \
        #         (not isinstance(update_mode['frequency'], int) or update_mode['frequency'] < 1):
        #     raise TensorforceError.value(
        #         name='update_mode', argument='frequency', value=update_mode['frequency']
        #     )
        # update_mode: start
        # elif 'start' in update_mode and \
        #         (not isinstance(update_mode['start'], int) or update_mode['start'] < 0):
        #     raise TensorforceError.value(
        #         name='update_mode', argument='start', value=update_mode['start']
        #     )
        self.update_mode = update_mode
        self.update_unit = update_mode['unit']
        self.update_batch_size = self.add_module(
            name='update-batch-size', module=update_mode['batch_size'], modules=parameter_modules,
            dtype='long'
        )
        self.update_frequency = self.add_module(
            name='update-frequency',
            module=update_mode.get('frequency', update_mode['batch_size']),
            modules=parameter_modules, dtype='long'
        )
        self.update_start = self.add_module(
            name='update-start', module=update_mode.get('frequency', 0), modules=parameter_modules,
            dtype='long'
        )
        if self.update_unit == 'sequences':
            self.update_sequence_length = self.add_module(
                name='update-sequence-length', module=update_mode['sequence_length'],
                modules=parameter_modules, dtype='long'
            )

        # Memory
        self.memory = self.add_module(
            name='memory', module=memory, modules=memory_modules, is_trainable=False,
            states_spec=self.states_spec, internals_spec=self.internals_spec,
            actions_spec=self.actions_spec
        )

        # Optimizer
        self.optimizer = self.add_module(
            name='optimizer', module=optimizer, modules=optimizer_modules
        )

        # Discount
        # if discount is not None and not isinstance(discount, dict) and \
        #         not isinstance(discount, float):
        #     raise TensorforceError.type(name='discount', value=discount)
        # elif discount is not None and not isinstance(variable_noise, dict) and discount < 0.0:
        #     raise TensorforceError.value(name='discount', value=discount)
        discount = 1.0 if discount is None else discount
        self.discount = self.add_module(
            name='discount', module=discount, modules=parameter_modules, dtype='float'
        )

    def as_local_model(self):
        """
        Makes sure our optimizer is wrapped into the global_optimizer meta. This is only relevant for distributed RL.
        """
        super().as_local_model()

        self.optimizer_spec = dict(
            type='global_optimizer',
            optimizer=self.optimizer_spec
        )

    #def tf_discounted_cumulative_reward(self, terminal, reward, discount, final_reward=0.0):
    #    """
    #    Creates the TensorFlow operations for calculating the discounted cumulative rewards
    #    for a given sequence of rewards.

    #    Args:
    #        terminal: Terminal boolean tensor.
    #        reward: Reward tensor.
    #        discount: Discount factor.
    #        final_reward: Last reward value in the sequence.

    #    Returns:
    #        Discounted cumulative reward tensor.
    #    """

    #    # TODO: n-step cumulative reward (particularly for envs without terminal)

    #    def cumulate(cumulative, reward_and_terminal):
    #        rew, term = reward_and_terminal
    #        return tf.where(condition=term, x=rew, y=(rew + cumulative * discount))

    #    # Reverse since reward cumulation is calculated right-to-left, but tf.scan only works left-to-right
    #    reward = tf.reverse(tensor=reward, axis=(0,))
    #    terminal = tf.reverse(tensor=terminal, axis=(0,))

    #    reward = tf.scan(fn=cumulate, elems=(reward, terminal), initializer=tf.stop_gradient(input=final_reward))

    #    return tf.reverse(tensor=reward, axis=(0,))

    # TODO: could be a utility helper function if we remove self.discount and only allow external discount-value input
    def tf_discounted_cumulative_reward(self, terminal, reward, discount=None, final_reward=0.0, horizon=0):
        """
        Creates and returns the TensorFlow operations for calculating the sequence of discounted cumulative rewards
        for a given sequence of single rewards.

        Example:
        single rewards = 2.0 1.0 0.0 0.5 1.0 -1.0
        terminal = False, False, False, False True False
        gamma = 0.95
        final_reward = 100.0 (only matters for last episode (r=-1.0) as this episode has no terminal signal)
        horizon=3
        output = 2.95 1.45 1.38 1.45 1.0 94.0

        Args:
            terminal: Tensor (bool) holding the is-terminal sequence. This sequence may contain more than one
                True value. If its very last element is False (not terminating), the given `final_reward` value
                is assumed to follow the last value in the single rewards sequence (see below).
            reward: Tensor (float) holding the sequence of single rewards. If the last element of `terminal` is False,
                an assumed last reward of the value of `final_reward` will be used.
            discount (float): The discount factor (gamma). By default, take the Model's discount factor.
            final_reward (float): Reward value to use if last episode in sequence does not terminate (terminal sequence
                ends with False). This value will be ignored if horizon == 1 or discount == 0.0.
            horizon (int): The length of the horizon (e.g. for n-step cumulative rewards in continuous tasks
                without terminal signals). Use 0 (default) for an infinite horizon. Note that horizon=1 leads to the
                exact same results as a discount factor of 0.0.

        Returns:
            Discounted cumulative reward tensor with the same shape as `reward`.
        """

        # By default -> take Model's gamma value
        if discount is None:
            discount = self.discount.value()

        # Accumulates discounted (n-step) reward (start new if terminal)
        def cumulate(cumulative, reward_terminal_horizon_subtract):
            rew, is_terminal, is_over_horizon, sub = reward_terminal_horizon_subtract
            return tf.where(
                # If terminal, start new cumulation.
                condition=is_terminal,
                x=rew,
                y=tf.where(
                    # If we are above the horizon length (H) -> subtract discounted value from H steps back.
                    condition=is_over_horizon,
                    x=(rew + cumulative * discount - sub),
                    y=(rew + cumulative * discount)
                )
            )

        # Accumulates length of episodes (starts new if terminal)
        def len_(cumulative, term):
            return tf.where(
                condition=term,
                # Start counting from 1 after is-terminal signal
                x=tf.ones_like(tensor=term, dtype=util.tf_dtype(dtype='int')),
                # Otherwise, increase length by 1
                y=(cumulative + tf.constant(value=1, dtype=util.tf_dtype(dtype='int')))
            )

        # Reverse, since reward cumulation is calculated right-to-left, but tf.scan only works left-to-right.
        reward = tf.reverse(tensor=reward, axis=(0,))
        # e.g. -1.0 1.0 0.5 0.0 1.0 2.0
        terminal = tf.reverse(tensor=terminal, axis=(0,))
        # e.g. F T F F F F

        # Store the steps until end of the episode(s) determined by the input terminal signals (True starts new count).
        lengths = tf.scan(
            fn=len_, elems=terminal,
            initializer=tf.zeros_like(tensor=terminal[0], dtype=util.tf_dtype(dtype='int'))
        )
        # e.g. 1 1 2 3 4 5
        off_horizon = tf.greater(x=lengths, y=tf.fill(dims=tf.shape(input=lengths), value=tf.constant(value=horizon, dtype=util.tf_dtype(dtype='int'))))
        # e.g. F F F F T T

        # Calculate the horizon-subtraction value for each step.
        if horizon > 0:
            horizon_subtractions = tf.map_fn(lambda x: (discount ** horizon) * x, reward, dtype=util.tf_dtype(dtype='float'))
            # Shift right by size of horizon (fill rest with 0.0).
            horizon_subtractions = tf.concat([np.zeros(shape=(horizon,), dtype=util.tf_dtype(dtype='float')), horizon_subtractions], axis=0)
            horizon_subtractions = tf.slice(horizon_subtractions, begin=(0,), size=tf.shape(reward))
            # e.g. 0.0, 0.0, 0.0, -1.0*g^3, 1.0*g^3, 0.5*g^3
        # all 0.0 if infinite horizon (special case: horizon=0)
        else:
            horizon_subtractions = tf.zeros(shape=tf.shape(reward), dtype=util.tf_dtype(dtype='float'))

        # Now do the scan, each time summing up the previous step (discounted by gamma) and
        # subtracting the respective `horizon_subtraction`.
        if isinstance(final_reward, float):
            final_reward = tf.constant(value=final_reward, dtype=util.tf_dtype(dtype='float'))
        reward = tf.scan(
            fn=cumulate,
            elems=(reward, terminal, off_horizon, horizon_subtractions),
            initializer=final_reward if horizon != 1 else tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        )
        # Re-reverse again to match input sequences.
        return tf.reverse(tensor=reward, axis=(0,))

    def tf_reference(
        self, states, internals, actions, terminal, reward, next_states, next_internals
    ):
        """
        Creates the TensorFlow operations for obtaining the reference tensor(s), in case of a
        comparative loss.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            next_states: Dict of successor state tensors.
            next_internals: List of posterior internal state tensors.

        Returns:
            Reference tensor(s).
        """
        return None

    def tf_loss_per_instance(
        self, states, internals, actions, terminal, reward, next_states, next_internals,
        reference=None
    ):
        """
        Creates the TensorFlow operations for calculating the loss per batch instance.

        Args:
            states: Dict of state tensors.
            internals: Dict of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            next_states: Dict of successor state tensors.
            next_internals: List of posterior internal state tensors.
            reference: Optional reference tensor(s), in case of a comparative loss.

        Returns:
            Loss per instance tensor.
        """
        raise NotImplementedError

    def tf_total_loss(
        self, states, internals, actions, terminal, reward, next_states, next_internals,
        reference=None
    ):
        """
        Creates the TensorFlow operations for calculating the full loss of a batch.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            next_states: Dict of successor state tensors.
            next_internals: List of posterior internal state tensors.
            reference: Optional reference tensor(s), in case of a comparative loss.

        Returns:
            Loss tensor.
        """
        # # Set global tensors
        # Module.update_tensors(**states, **internals, **actions, terminal=terminal, reward=reward)

        # Mean loss per instance
        loss_per_instance = self.loss_per_instance(
            states=states, internals=internals, actions=actions, terminal=terminal, reward=reward,
            next_states=next_states, next_internals=next_internals, reference=reference
        )

        # Returns no-op
        updated = self.memory.update_batch(loss_per_instance=loss_per_instance)
        with tf.control_dependencies(control_inputs=(updated,)):
            loss = tf.math.reduce_mean(input_tensor=loss_per_instance, axis=0)
            loss = self.add_summary(
                label=('objective-loss', 'losses'), name='objective-loss', tensor=loss
            )

        # Regularization losses
        reg_loss = self.regularize(states=states, internals=internals)
        reg_loss = self.add_summary(
            label=('regularization-loss', 'losses'), name='regularization-loss', tensor=reg_loss
        )

        loss = loss + reg_loss
        loss = self.add_summary(label=('loss', 'losses'), name='loss', tensor=loss)

        return loss

    def optimizer_arguments(
        self, states, internals, actions, terminal, reward, next_states, next_internals
    ):
        """
        Returns the optimizer arguments including the time, the list of variables to optimize,
        and various functions which the optimizer might require to perform an update step.

        Args:
            states (dict): Dict of state tensors.
            internals (dict): Dict of prior internal state tensors.
            actions (dict): Dict of action tensors.
            terminal: 1D boolean is-terminal tensor.
            reward: 1D (float) rewards tensor.
            next_states (dict): Dict of successor state tensors.
            next_internals (dict): Dict of posterior internal state tensors.

        Returns:
            Optimizer arguments as dict to be used as **kwargs to the optimizer.
        """
        arguments = dict(
            variables=self.get_variables(only_trainable=True),
            arguments=dict(
                states=states, internals=internals, actions=actions, terminal=terminal,
                reward=reward, next_states=next_states, next_internals=next_internals
            ),
            fn_reference=self.reference, fn_loss=self.total_loss
        )
        if self.global_model is not None:
            arguments['global_variables'] = self.global_model.get_variables(only_trainable=True)
        return arguments

    def tf_optimization(
        self, states, internals, actions, terminal, reward, next_states=None, next_internals=None
    ):
        """
        Creates the TensorFlow operations for performing an optimization update step based
        on the given input states and actions batch.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            next_states: Dict of successor state tensors.
            next_internals: List of posterior internal state tensors.

        Returns:
            The optimization operation.
        """
        arguments = self.optimizer_arguments(
            states=states, internals=internals, actions=actions, terminal=terminal, reward=reward,
            next_states=next_states, next_internals=next_internals
        )
        optimized = self.optimizer.minimize(**arguments)
        return optimized

    def tf_core_observe(self, states, internals, actions, terminal, reward):
        """
        Creates and returns the op that - if frequency condition is hit - pulls a batch from the memory
        and does one optimization step.
        """
        # Store timestep in memory
        stored = self.memory.store(
            states=states, internals=internals, actions=actions, terminal=terminal, reward=reward
        )

        # Periodic optimization
        with tf.control_dependencies(control_inputs=(stored,)):
            batch_size = self.update_batch_size.value()
            frequency = self.update_frequency.value()
            start = self.update_start.value()
            start = tf.maximum(x=start, y=batch_size)

            if self.update_unit == 'timesteps':
                # Timestep-based batch
                timestep = Module.retrieve_tensor(name='timestep')
                is_frequency = tf.math.equal(
                    x=tf.mod(x=timestep, y=frequency),
                    y=tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
                )
                at_least_start = tf.math.greater_equal(x=timestep, y=start)

            elif self.update_unit == 'sequences':
                # Timestep-sequence-based batch
                timestep = Module.retrieve_tensor(name='timestep')
                sequence_length = self.update_sequence_length.value()
                is_frequency = tf.math.equal(
                    x=tf.mod(x=timestep, y=frequency),
                    y=tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
                )
                at_least_start = tf.math.greater_equal(
                    x=timestep, y=(start + sequence_length - 1)
                )

            elif self.update_unit == 'episodes':
                # Episode-based batch
                episode = Module.retrieve_tensor(name='episode')
                is_frequency = tf.math.equal(
                    x=tf.mod(x=episode, y=frequency),
                    y=tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
                )
                # Only update once per episode increment
                is_frequency = tf.math.logical_and(
                    x=is_frequency, y=tf.reduce_any(input_tensor=terminal)
                )
                at_least_start = tf.math.greater_equal(x=episode, y=start)

            def optimize():
                if self.update_unit == 'timesteps':
                    # Timestep-based batch
                    batch = self.memory.retrieve_timesteps(n=batch_size)
                elif self.update_unit == 'episodes':
                    # Episode-based batch
                    batch = self.memory.retrieve_episodes(n=batch_size)
                elif self.update_unit == 'sequences':
                    # Timestep-sequence-based batch
                    batch = self.memory.retrieve_sequences(
                        n=batch_size, sequence_length=sequence_length
                    )

                # Do not calculate gradients for memory-internal operations.
                batch = util.fmap(function=tf.stop_gradient, xs=batch)
                Module.update_tensors(
                    update=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
                )
                optimized = self.optimization(**batch)

                return optimized

            do_optimize = tf.math.logical_and(x=is_frequency, y=at_least_start)

            optimized = self.cond(pred=do_optimize, true_fn=optimize, false_fn=util.no_operation)

            return optimized

    def tf_import_experience(self, states, internals, actions, terminal, reward):
        """
        Imports experiences into the TensorFlow memory structure. Can be used to import
        off-policy data.

        :param states: Dict of state values to import with keys as state names and values as values to set.
        :param internals: Internal values to set, can be fetched from agent via agent.current_internals
            if no values available.
        :param actions: Dict of action values to import with keys as action names and values as values to set.
        :param terminal: Terminal value(s)
        :param reward: Reward value(s)
        """
        return self.memory.store(
            states=states, internals=internals, actions=actions, terminal=terminal, reward=reward
        )

    def create_operations(self, states, internals, actions, terminal, reward, deterministic, independent, parallel):
        # Import experience operation.
        # self.import_experience_output = self.fn_import_experience(
        #     states=states, internals=internals, actions=actions, terminal=terminal, reward=reward
        # )

        super().create_operations(
            states=states, internals=internals, actions=actions, terminal=terminal, reward=reward,
            deterministic=deterministic, independent=independent, parallel=parallel
        )

    # def import_experience(self, states, internals, actions, terminal, reward):
    #     """
    #     Stores experiences.
    #     """
    #     fetches = self.import_experience_output

    #     feed_dict = self.get_feed_dict(
    #         states=states,
    #         internals=internals,
    #         actions=actions,
    #         terminal=terminal,
    #         reward=reward
    #     )

    #     self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)
