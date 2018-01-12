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

"""
???
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce import TensorForceError
from tensorforce.core.memories import Memory
from tensorforce.core.optimizers import Optimizer
from tensorforce.models import Model


class MemoryModel(Model):
    """
    ???
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        device,
        session_config,
        scope,
        saver_spec,
        summary_spec,
        distributed_spec,
        variable_noise,
        states_preprocessing,
        actions_exploration,
        reward_preprocessing,
        memory,
        update_spec,
        optimizer,
        discount
    ):
        self.memory = memory
        self.update_spec = update_spec
        self.optimizer = optimizer
        self.discount = discount

        super(MemoryModel, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            device=device,
            session_config=session_config,
            scope=scope,
            saver_spec=saver_spec,
            summary_spec=summary_spec,
            distributed_spec=distributed_spec,
            variable_noise=variable_noise,
            states_preprocessing=states_preprocessing,
            actions_exploration=actions_exploration,
            reward_preprocessing=reward_preprocessing
        )

    def initialize(self, custom_getter):
        super(MemoryModel, self).initialize(custom_getter)

        # Memory
        self.memory = Memory.from_spec(
            spec=self.memory,
            kwargs=dict(
                states_spec=self.states_spec,
                actions_spec=self.actions_spec,
                include_next_states=False,  # !!!!!!!!!!!!!!!!!!!!!!!!!!!
                summary_labels=self.summary_labels
            )
        )
        self.memory.initialize()

        # Optimizer
        if self.optimizer is not None:
            self.optimizer = Optimizer.from_spec(
                spec=self.optimizer,
                kwargs=dict(
                    summaries=self.summaries,
                    summary_labels=self.summary_labels
                )
            )

        # TensorFlow functions
        self.fn_discounted_cumulative_reward = tf.make_template(
            name_=(self.scope + '/discounted-cumulative-reward'),
            func_=self.tf_discounted_cumulative_reward,
            custom_getter_=custom_getter
        )
        self.fn_loss_per_instance = tf.make_template(
            name_=(self.scope + '/loss-per-instance'),
            func_=self.tf_loss_per_instance,
            custom_getter_=custom_getter
        )
        self.fn_regularization_losses = tf.make_template(
            name_=(self.scope + '/regularization-losses'),
            func_=self.tf_regularization_losses,
            custom_getter_=custom_getter
        )
        self.fn_loss = tf.make_template(
            name_=(self.scope + '/loss'),
            func_=self.tf_loss,
            custom_getter_=custom_getter
        )
        self.fn_optimization = tf.make_template(
            name_=(self.scope + '/optimization'),
            func_=self.tf_optimization,
            custom_getter_=custom_getter
        )

    def tf_discounted_cumulative_reward(self, terminal, reward, discount, final_reward=0.0):
        """
        Creates the TensorFlow operations for calculating the discounted cumulative rewards
        for a given sequence of rewards.

        Args:
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            discount: Discount factor.
            final_reward: Last reward value in the sequence.

        Returns:
            Discounted cumulative reward tensor.
        """

        # TODO: n-step cumulative reward (particularly for envs without terminal)

        def cumulate(cumulative, reward_and_terminal):
            rew, term = reward_and_terminal
            return tf.where(condition=term, x=rew, y=(rew + cumulative * discount))

        # Reverse since reward cumulation is calculated right-to-left, but tf.scan only works left-to-right
        reward = tf.reverse(tensor=reward, axis=(0,))
        terminal = tf.reverse(tensor=terminal, axis=(0,))

        reward = tf.scan(fn=cumulate, elems=(reward, terminal), initializer=final_reward)

        return tf.reverse(tensor=reward, axis=(0,))

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward, next_states, next_internals, update):
        """
        Creates the TensorFlow operations for calculating the loss per batch instance.

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
            Loss tensor.
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

    def tf_loss(self, states, internals, actions, terminal, reward, next_states, next_internals, update):
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
            update=update
        )
        loss = tf.reduce_mean(input_tensor=loss_per_instance, axis=0)

        # Loss without regularization summary
        if 'losses' in self.summary_labels:
            summary = tf.summary.scalar(name='loss-without-regularization', tensor=loss)
            self.summaries.append(summary)

        # Regularization losses
        losses = self.fn_regularization_losses(states=states, internals=internals, update=update)
        if len(losses) > 0:
            loss += tf.add_n(inputs=list(losses.values()))
            if 'regularization' in self.summary_labels:
                for name, loss_val in losses.items():
                    summary = tf.summary.scalar(name=('regularization/' + name), tensor=loss_val)
                    self.summaries.append(summary)

        # Total loss summary
        if 'losses' in self.summary_labels or 'total-loss' in self.summary_labels:
            summary = tf.summary.scalar(name='total-loss', tensor=loss)
            self.summaries.append(summary)

        return loss

    def optimizer_arguments(self, states, internals, actions, terminal, reward, next_states, next_internals):
        """
        Returns the optimizer arguments including the time, the list of variables to optimize,
        and various argument-free functions (in particular `fn_loss` returning the combined
        0-dim batch loss tensor) which the optimizer might require to perform an update step.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            next_states: Dict of successor state tensors.
            next_internals: List of posterior internal state tensors.

        Returns:
            Optimizer arguments as dict.
        """
        arguments = dict()
        arguments['time'] = self.timestep
        arguments['variables'] = self.get_variables()
        arguments['states'] = states
        arguments['internals'] = internals
        arguments['actions'] = actions
        arguments['terminal'] = terminal
        arguments['reward'] = reward
        arguments['next_states'] = next_states
        arguments['next_internals'] = next_internals
        arguments['update'] = tf.constant(value=False)
        arguments['fn_loss'] = self.fn_loss
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
        return self.optimizer.minimize(**self.optimizer_arguments(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            next_states=next_states,
            next_internals=next_internals
        ))

    def tf_observe_timestep(self, states, internals, actions, terminal, reward):
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
            mode = self.update_spec['mode']
            batch_size = self.update_spec['batch_size']
            frequency = self.update_spec['frequency']

            if mode == 'timesteps':
                optimize = tf.logical_and(
                    x=tf.equal(x=(self.timestep % frequency), y=0),
                    y=tf.greater_equal(x=self.timestep, y=batch_size)
                )
                batch = self.memory.retrieve_timesteps(n=batch_size)

            elif mode == 'episodes':
                optimize = tf.logical_and(
                    x=tf.equal(x=(self.episode % frequency), y=0),
                    y=tf.logical_and(
                        # Only update once per episode increment.
                        x=tf.greater(x=tf.count_nonzero(input_tensor=terminal), y=0),
                        y=tf.greater_equal(x=self.episode, y=batch_size)
                    )
                )
                batch = self.memory.retrieve_episodes(n=batch_size)

            elif mode == 'sequences':
                optimize = tf.logical_and(
                    x=tf.equal(x=(self.timestep % frequency), y=0),
                    y=tf.greater_equal(x=self.timestep, y=batch_size)
                )
                batch = self.memory.retrieve_sequences(n=batch_size)

            else:
                raise TensorForceError("Invalid update mode: {}.".format(mode))

            # optimize = tf.Print(optimize, (optimize,))
            optimization = tf.cond(
                pred=optimize,
                true_fn=(lambda: self.tf_optimization(**batch)),
                false_fn=tf.no_op
            )

        return optimization

    def get_variables(self, include_non_trainable=False):
        """
        Returns the TensorFlow variables used by the model.

        Returns:
            List of variables.
        """
        model_variables = super(MemoryModel, self).get_variables(include_non_trainable=include_non_trainable)

        if include_non_trainable:
            memory_variables = self.memory.get_variables()
            optimizer_variables = self.optimizer.get_variables()
            return model_variables + memory_variables + optimizer_variables

        else:
            return model_variables

    def get_summaries(self):
        model_summaries = super(MemoryModel, self).get_summaries()
        memory_summaries = self.network.get_summaries()

        return model_summaries + memory_summaries
