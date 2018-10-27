# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce import util, TensorForceError
from tensorforce.core.memories import Memory
from tensorforce.core.optimizers import Optimizer
from tensorforce.models import Model


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
        states,
        actions,
        scope,
        device,
        saver,
        summarizer,
        execution,
        batching_capacity,
        variable_noise,
        states_preprocessing,
        actions_exploration,
        reward_preprocessing,
        update_mode,
        memory,
        optimizer,
        discount
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
        self.update_mode = update_mode
        self.memory_spec = memory
        self.optimizer_spec = optimizer

        # Discount
        assert discount is None or discount >= 0.0
        self.discount = discount

        self.memory = None
        self.optimizer = None

        self.fn_discounted_cumulative_reward = None
        self.fn_reference = None
        self.fn_loss_per_instance = None
        self.fn_regularization_losses = None
        self.fn_loss = None
        self.fn_optimization = None
        self.fn_import_experience = None

        super(MemoryModel, self).__init__(
            states=states,
            actions=actions,
            scope=scope,
            device=device,
            saver=saver,
            summarizer=summarizer,
            execution=execution,
            batching_capacity=batching_capacity,
            variable_noise=variable_noise,
            states_preprocessing=states_preprocessing,
            actions_exploration=actions_exploration,
            reward_preprocessing=reward_preprocessing
        )

    def as_local_model(self):
        """
        Makes sure our optimizer is wrapped into the global_optimizer meta. This is only relevant for distributed RL.
        """
        super(MemoryModel, self).as_local_model()
        self.optimizer_spec = dict(
            type='global_optimizer',
            optimizer=self.optimizer_spec
        )

    def setup_components_and_tf_funcs(self, custom_getter=None):
        """
        Constructs the memory and the optimizer objects.
        Generates and stores all template functions.
        """
        custom_getter = super(MemoryModel, self).setup_components_and_tf_funcs(custom_getter)

        # Memory
        self.memory = Memory.from_spec(
            spec=self.memory_spec,
            kwargs=dict(
                states=self.states_spec,
                internals=self.internals_spec,
                actions=self.actions_spec,
                summary_labels=self.summary_labels
            )
        )

        # Optimizer
        self.optimizer = Optimizer.from_spec(
            spec=self.optimizer_spec,
            kwargs=dict(summary_labels=self.summary_labels)
        )

        # TensorFlow functions
        self.fn_discounted_cumulative_reward = tf.make_template(
            name_='discounted-cumulative-reward',
            func_=self.tf_discounted_cumulative_reward,
            custom_getter_=custom_getter
        )
        self.fn_reference = tf.make_template(
            name_='reference',
            func_=self.tf_reference,
            custom_getter_=custom_getter
        )
        self.fn_loss_per_instance = tf.make_template(
            name_='loss-per-instance',
            func_=self.tf_loss_per_instance,
            custom_getter_=custom_getter
        )
        self.fn_regularization_losses = tf.make_template(
            name_='regularization-losses',
            func_=self.tf_regularization_losses,
            custom_getter_=custom_getter
        )
        self.fn_loss = tf.make_template(
            name_='loss',
            func_=self.tf_loss,
            custom_getter_=custom_getter
        )
        self.fn_optimization = tf.make_template(
            name_='optimization',
            func_=self.tf_optimization,
            custom_getter_=custom_getter
        )
        self.fn_import_experience = tf.make_template(
            name_='import-experience',
            func_=self.tf_import_experience,
            custom_getter_=custom_getter
        )

        return custom_getter

    def tf_initialize(self):
        """
        Also initializes our Memory object (self.memory).
        """
        super(MemoryModel, self).tf_initialize()
        self.memory.initialize()

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
            discount = self.discount

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
                x=tf.ones(shape=(), dtype=tf.int32),
                # Otherwise, increase length by 1
                y=cumulative + 1
            )

        # Reverse, since reward cumulation is calculated right-to-left, but tf.scan only works left-to-right.
        reward = tf.reverse(tensor=reward, axis=(0,))
        # e.g. -1.0 1.0 0.5 0.0 1.0 2.0
        terminal = tf.reverse(tensor=terminal, axis=(0,))
        # e.g. F T F F F F

        # Store the steps until end of the episode(s) determined by the input terminal signals (True starts new count).
        lengths = tf.scan(fn=len_, elems=terminal, initializer=0)
        # e.g. 1 1 2 3 4 5
        off_horizon = tf.greater(lengths, tf.fill(dims=tf.shape(lengths), value=horizon))
        # e.g. F F F F T T

        # Calculate the horizon-subtraction value for each step.
        if horizon > 0:
            horizon_subtractions = tf.map_fn(lambda x: (discount ** horizon) * x, reward, dtype=tf.float32)
            # Shift right by size of horizon (fill rest with 0.0).
            horizon_subtractions = tf.concat([np.zeros(shape=(horizon,)), horizon_subtractions], axis=0)
            horizon_subtractions = tf.slice(horizon_subtractions, begin=(0,), size=tf.shape(reward))
            # e.g. 0.0, 0.0, 0.0, -1.0*g^3, 1.0*g^3, 0.5*g^3
        # all 0.0 if infinite horizon (special case: horizon=0)
        else:
            horizon_subtractions = tf.zeros(shape=tf.shape(reward))

        # Now do the scan, each time summing up the previous step (discounted by gamma) and
        # subtracting the respective `horizon_subtraction`.
        reward = tf.scan(
            fn=cumulate,
            elems=(reward, terminal, off_horizon, horizon_subtractions),
            initializer=final_reward if horizon != 1 else 0.0
        )
        # Re-reverse again to match input sequences.
        return tf.reverse(tensor=reward, axis=(0,))

    def tf_reference(self, states, internals, actions, terminal, reward, next_states, next_internals, update):
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
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            Reference tensor(s).
        """
        return None

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward,
                             next_states, next_internals, update, reference=None):
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
            update: Boolean tensor indicating whether this call happens during an update.
            reference: Optional reference tensor(s), in case of a comparative loss.

        Returns:
            Loss per instance tensor.
        """
        raise NotImplementedError

    def tf_regularization_losses(self, states, internals, update):
        """
        Creates the TensorFlow operations for calculating the regularization losses for the given input states.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            Dict of regularization loss tensors.
        """
        return dict()

    def tf_loss(self, states, internals, actions, terminal, reward, next_states, next_internals, update, reference=None):
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
            update: Boolean tensor indicating whether this call happens during an update.
            reference: Optional reference tensor(s), in case of a comparative loss.

        Returns:
            Loss tensor.
        """
        # Mean loss per instance
        loss_per_instance = self.fn_loss_per_instance(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            next_states=next_states,
            next_internals=next_internals,
            update=update,
            reference=reference
        )

        # Returns no-op.
        updated = self.memory.update_batch(loss_per_instance=loss_per_instance)
        with tf.control_dependencies(control_inputs=(updated,)):
            loss = tf.reduce_mean(input_tensor=loss_per_instance, axis=0)

            # Loss without regularization summary.
            if 'losses' in self.summary_labels:
                tf.contrib.summary.scalar(name='loss-without-regularization', tensor=loss)

            # Regularization losses.
            losses = self.fn_regularization_losses(states=states, internals=internals, update=update)
            if len(losses) > 0:
                loss += tf.add_n(inputs=[losses[name] for name in sorted(losses)])
                if 'regularization' in self.summary_labels:
                    for name in sorted(losses):
                        tf.contrib.summary.scalar(name=('regularization/' + name), tensor=losses[name])

            # Total loss summary.
            if 'losses' in self.summary_labels or 'total-loss' in self.summary_labels:
                tf.contrib.summary.scalar(name='total-loss', tensor=loss)

            return loss

    def optimizer_arguments(self, states, internals, actions, terminal, reward, next_states, next_internals):
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
            time=self.global_timestep,
            variables=self.get_variables(),
            arguments=dict(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward,
                next_states=next_states,
                next_internals=next_internals,
                update=tf.constant(value=True)
            ),
            fn_reference=self.fn_reference,
            fn_loss=self.fn_loss
        )
        if self.global_model is not None:
            arguments['global_variables'] = self.global_model.get_variables()
        return arguments

    def tf_optimization(self, states, internals, actions, terminal, reward, next_states=None, next_internals=None):
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
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            next_states=next_states,
            next_internals=next_internals
        )
        return self.optimizer.minimize(**arguments)

    def tf_observe_timestep(self, states, internals, actions, terminal, reward):
        """
        Creates and returns the op that - if frequency condition is hit - pulls a batch from the memory
        and does one optimization step.
        """
        # Store timestep in memory
        stored = self.memory.store(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )

        # Periodic optimization
        with tf.control_dependencies(control_inputs=(stored,)):
            unit = self.update_mode['unit']
            batch_size = self.update_mode['batch_size']
            frequency = self.update_mode.get('frequency', batch_size)
            first_update = self.update_mode.get('first_update', 0)

            if unit == 'timesteps':
                # Timestep-based batch
                optimize = tf.logical_and(
                    x=tf.equal(x=(self.timestep % frequency), y=0),
                    y=tf.logical_and(
                        x=tf.greater_equal(x=self.timestep, y=batch_size),
                        y=tf.greater_equal(x=self.timestep, y=first_update)
                    )
                )

            elif unit == 'episodes':
                # Episode-based batch
                optimize = tf.logical_and(
                    x=tf.equal(x=(self.episode % frequency), y=0),
                    y=tf.logical_and(
                        # Only update once per episode increment.
                        x=tf.greater(x=tf.count_nonzero(input_tensor=terminal), y=0),
                        y=tf.logical_and(
                            x=tf.greater_equal(x=self.episode, y=batch_size),
                            y=tf.greater_equal(x=self.episode, y=first_update)
                        )
                    )
                )

            elif unit == 'sequences':
                # Timestep-sequence-based batch
                sequence_length = self.update_mode.get('length', 8)
                optimize = tf.logical_and(
                    x=tf.equal(x=(self.timestep % frequency), y=0),
                    y=tf.logical_and(
                        x=tf.greater_equal(x=self.timestep, y=(batch_size + sequence_length - 1)),
                        y=tf.greater_equal(x=self.timestep, y=first_update)
                    )
                )

            else:
                raise TensorForceError("Invalid update unit: {}.".format(unit))

            def true_fn():
                if unit == 'timesteps':
                    # Timestep-based batch
                    batch = self.memory.retrieve_timesteps(n=batch_size)
                elif unit == 'episodes':
                    # Episode-based batch
                    batch = self.memory.retrieve_episodes(n=batch_size)
                elif unit == 'sequences':
                    # Timestep-sequence-based batch
                    batch = self.memory.retrieve_sequences(n=batch_size, sequence_length=sequence_length)

                # Do not calculate gradients for memory-internal operations.
                batch = util.map_tensors(
                    fn=(lambda tensor: tf.stop_gradient(input=tensor)),
                    tensors=batch
                )

                optimize = self.fn_optimization(**batch)
                with tf.control_dependencies(control_inputs=(optimize,)):
                    return tf.logical_and(x=True, y=True)

            return tf.cond(pred=optimize, true_fn=true_fn, false_fn=tf.no_op)

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
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )

    def create_operations(self, states, internals, actions, terminal, reward, deterministic, independent, index):
        # Import experience operation.
        self.import_experience_output = self.fn_import_experience(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )

        super(MemoryModel, self).create_operations(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            deterministic=deterministic,
            independent=independent,
            index=index
        )

    def get_variables(self, include_submodules=False, include_nontrainable=False):
        model_variables = super(MemoryModel, self).get_variables(
            include_submodules=include_submodules,
            include_nontrainable=include_nontrainable
        )

        if include_nontrainable:
            memory_variables = self.memory.get_variables()
            model_variables += memory_variables

            optimizer_variables = self.optimizer.get_variables()
            # For some reason, some optimizer variables are only registered in the model.
            for variable in optimizer_variables:
                if variable in model_variables:
                    model_variables.remove(variable)
            model_variables += optimizer_variables

        return model_variables

    def import_experience(self, states, internals, actions, terminal, reward):
        """
        Stores experiences.
        """
        fetches = self.import_experience_output

        feed_dict = self.get_feed_dict(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )

        self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)
