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
The `Model` class coordinates the creation and execution of all TensorFlow operations within a model.
It implements the `reset`, `act` and `update` functions, which give the interface the `Agent` class
communicates with, and which should not need to be overwritten. Instead, the following TensorFlow
functions need to be implemented:

* `tf_actions_and_internals(states, internals, deterministic)` returning the batch of
   actions and successor internal states.
* `tf_loss_per_instance(states, internals, actions, terminal, reward)` returning the loss
   per instance for a batch.

Further, the following TensorFlow functions should be extended accordingly:

* `initialize(custom_getter)` defining TensorFlow placeholders/functions and adding internal states.
* `get_variables()` returning the list of TensorFlow variables (to be optimized) of this model.
* `tf_regularization_losses(states, internals)` returning a dict of regularization losses.
* `get_optimizer_kwargs(states, internals, actions, terminal, reward)` returning a dict of potential
   arguments (argument-free functions) to the optimizer.

Finally, the following TensorFlow functions can be useful in some cases:

* `get_states(states)` for state preprocessing, returning the processed batch of states.
* `get_actions(actions)` for action preprocessing, returning the processed batch of actions.
* `get_reward(states, internals, terminal, reward)` for reward preprocessing (like reward normalization), returning the processed batch of rewards.
* `create_output_operations(states, internals, actions, terminal, reward, deterministic)` for further output operations, similar to the two above for `Model.act` and `Model.update`.
* `tf_optimization(states, internals, actions, terminal, reward)` for further optimization operations (like the baseline update in a `PGModel` or the target network update in a `QModel`), returning a single grouped optimization operation.
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce import TensorForceError, util
from tensorforce.core.optimizers import Optimizer


class Model(object):
    """
    Base class for all (TensorFlow-based) models
    """


    # default_config = dict(
    #     discount=0.97,
    #     optimizer=dict(
    #         type='adam',
    #         learning_rate=0.0001
    #     ),
    #     device=None,
    #     tf_summary=None,
    #     tf_summary_level=0,
    #     tf_summary_interval=1000,
    #     distributed=False,
    #     global_model=False,
    #     session=None
    # )

    def __init__(self, states_spec, actions_spec, config):
        # States and actions specifications
        self.states_spec = states_spec
        self.actions_spec = actions_spec

        # Discount factor
        self.discount = config.discount

        # Reward normalization
        assert isinstance(config.normalize_rewards, bool)
        self.normalize_rewards = config.normalize_rewards

        # TODO: Move logging to Agent? Since Model is pure TensorFlow
        # self.logger = logging.getLogger(self.__class__.__name__)
        # self.logger.setLevel(util.log_levels[config.log_level])

        # TensorFlow summaries
        self.summary_labels = set(config.summary_labels or ())
        self.summary_frequency = config.summary_frequency
        self.last_summary = -self.summary_frequency

        self.distributed = config.distributed

        if config.distributed:
            # Distributed model
            if config.global_model:
                # Global and local model for asynchronous updates
                global_default = dict(
                    # scope='global',
                    distributed = False,
                    global_model=True,
                    device=tf.train.replica_device_setter(
                        ps_tasks=1,
                        worker_device=config.device,
                        cluster=config.cluster_spec
                    )
                )
                global_config = config.copy()
                global_config.default(global_default)

                self.global_model = self.__class__(config=global_config)  # states_spec, actions_spec, config)
                self.global_timestep = self.global_model.global_timestep
                self.global_episode = self.global_model.episode
                # self.global_variables = self.global_model.variables

        else:
            pass
            # No distributed model
            # self.session = tf.Session()
            # self.session.reset()

        with tf.device(device_name_or_function=config.device):  # TODO: config.device!!!

            if config.distributed and config.global_model:
                pass
                # general self.time ???
                # self.global_timestep = tf.get_variable(name='timestep', dtype=tf.int32, initializer=0, trainable=False)
                # Problem how to record episode for MemoryAgent?
                # self.episode = tf.get_variable(name='episode', dtype=tf.int32, initializer=0, trainable=False)

            self.variables = dict()
            self.summaries = list()

            with tf.name_scope(name=config.scope):
                def custom_getter(getter, name, *args, **kwargs):
                    variable = getter(name=name, *args, **kwargs)
                    if not name.startswith('optimization'):
                        self.variables[name] = variable
                    if 'variables' in self.summary_labels:
                        summary = tf.summary.histogram(name=name, values=variable)
                        self.summaries.append(summary)
                    return variable

                # Create placeholders, tf functions, internals, etc
                self.initialize(custom_getter=custom_getter)

                # Input tensors
                states = self.get_states(states=self.state_inputs)
                internals = [tf.identity(input=internal) for internal in self.internal_inputs]
                actions = self.get_actions(actions=self.action_inputs)
                terminal = tf.identity(input=self.terminal_input)
                reward = self.get_reward(states=states, internals=internals, terminal=terminal, reward=self.reward_input)

                # Stop gradients for input preprocessing
                states = {name: tf.stop_gradient(input=state) for name, state in states.items()}
                actions = {name: tf.stop_gradient(input=action) for name, action in actions.items()}
                reward = tf.stop_gradient(input=reward)

                # Optimizer
                if not config.distributed or config.global_model:
                    self.optimizer = Optimizer.from_spec(spec=config.optimizer)
                else:
                    self.optimizer = GlobalOptimizer(optimizer=config.optimizer)

                # Create output fetch operations
                self.create_output_operations(
                    states=states,
                    internals=internals,
                    actions=actions,
                    terminal=terminal,
                    reward=reward,
                    deterministic=self.deterministic
                )
                # TODO: if global_config.global_model == True, then no optimization stuff


                if config.distributed and not config.global_model:
                    self.loss = tf.add_n(inputs=tf.losses.get_losses(scope=scope.name))
                    local_grads_and_vars = self.optimizer.compute_gradients(loss=self.loss, var_list=self.variables)
                    local_gradients = [grad for grad, var in local_grads_and_vars]
                    global_gradients = list(zip(local_gradients, self.global_model.variables))
                    self.update_local = tf.group(*(v1.assign(v2) for v1, v2 in zip(self.variables, self.global_model.variables)))
                    self.optimize = tf.group(
                        self.optimizer.apply_gradients(grads_and_vars=global_gradients),
                        self.update_local,
                        self.global_timestep.assign_add(tf.shape(self.reward)[0]))
                    self.increment_global_episode = self.global_episode.assign_add(tf.count_nonzero(input_tensor=self.terminal, dtype=tf.int32))

            # if config.distributed:
            #     scope_context.__exit__(None, None, None)

        self.saver = tf.train.Saver()

        # Initialize variables and finalize graph
        if not config.distributed:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            # self.session.graph.finalize()

        if config.summary_logdir is None:
            self.summary_writer = None
        else:
            self.summary_writer = tf.summary.FileWriter(logdir=config.summary_logdir, graph=self.session.graph)

    def initialize(self, custom_getter):
        """
        Creates the TensorFlow placeholders and functions for this model. Moreover adds the internal state
        placeholders and initialization values to the model.

        Args:
            custom_getter: The `custom_getter_` object to use for `tf.make_template` when creating TensorFlow functions.
        """
        # States
        self.state_inputs = dict()
        for name, state in self.states_spec.items():
            self.state_inputs[name] = tf.placeholder(
                dtype=util.tf_dtype(state['type']),
                shape=(None,) + tuple(state['shape']),
                name=name
            )

        # Actions
        self.action_inputs = dict()
        for name, action in self.actions_spec.items():
            self.action_inputs[name] = tf.placeholder(
                dtype=util.tf_dtype(action['type']),
                shape=(None,) + tuple(action['shape']),
                name=name
            )

        # Terminal
        self.terminal_input = tf.placeholder(dtype=tf.bool, shape=(None,), name='terminal')

        # Reward
        self.reward_input = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')

        # Internal states
        self.internal_inputs = list()
        self.internal_inits = list()

        # Deterministic action flag
        self.deterministic = tf.placeholder(dtype=tf.bool, shape=(), name='deterministic')

        # Timestep and episode
        # TODO: various modes !!!
        self.timestep = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
        # Problem how to record episode for MemoryAgent?
        # self.episode = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

        # TensorFlow functions
        self.fn_discounted_cumulative_reward = tf.make_template(
            name_=('discounted-cumulative-reward'),
            func_=self.tf_discounted_cumulative_reward,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )
        self.fn_actions_and_internals = tf.make_template(
            name_='actions-and-internals',
            func_=self.tf_actions_and_internals,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )
        self.fn_loss_per_instance = tf.make_template(
            name_='loss-per-instance',
            func_=self.tf_loss_per_instance,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )
        self.fn_regularization_losses = tf.make_template(
            name_='regularization-losses',
            func_=self.tf_regularization_losses,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )
        self.fn_loss = tf.make_template(
            name_='loss',
            func_=self.tf_loss,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )
        self.fn_optimization = tf.make_template(
            name_='optimization',
            func_=self.tf_optimization,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )
        self.fn_summarization = tf.make_template(
            name_='summarization',
            func_=self.tf_summarization,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )

    def get_states(self, states):
        # TODO: preprocessing could go here?
        return {name: tf.identity(input=state) for name, state in states.items()}

    def get_actions(self, actions):
        # TODO: preprocessing could go here?
        return {name: tf.identity(input=action) for name, action in actions.items()}

    def get_reward(self, states, internals, terminal, reward):
        if self.normalize_rewards:
            mean, variance = tf.nn.moments(x=reward, axes=0)
            return (reward - mean) / tf.maximum(x=variance, y=util.epsilon)
        else:
            return tf.identity(input=reward)

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

        def fn_scan(cumulative, reward_and_terminal):
            rew, term = reward_and_terminal
            return tf.cond(
                pred=term,
                true_fn=(lambda: rew),
                false_fn=(lambda: rew + cumulative * discount)
            )

        reward = tf.reverse(tensor=reward, axis=(0,))
        terminal = tf.reverse(tensor=terminal, axis=(0,))
        reward = tf.scan(fn=fn_scan, elems=(reward, terminal), initializer=final_reward)
        return tf.reverse(tensor=reward, axis=(0,))

    def tf_actions_and_internals(self, states, internals, deterministic):
        """
        Creates the TensorFlow operations for retrieving the actions (and posterior internal states)
        in reaction to the given input states (and prior internal states).

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            deterministic: If true, the action is chosen deterministically.

        Returns:
            Actions and list of posterior internal state tensors.
        """
        raise NotImplementedError

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward):
        """
        Creates the TensorFlow operations for calculating the loss per batch instance
        of the given input states and actions.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.

        Returns:
            Loss tensor.
        """
        raise NotImplementedError

    def tf_regularization_losses(self, states, internals):
        """
        Creates the TensorFlow operations for calculating the regularization losses for the given input states.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.

        Returns:
            Dict of regularization loss tensors.
        """
        return dict()

    def tf_loss(self, states, internals, actions, terminal, reward):
        # Mean loss per instance
        loss_per_instance = self.fn_loss_per_instance(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )
        loss = tf.reduce_mean(input_tensor=loss_per_instance, axis=0)

        # Loss without regularization summary
        if 'losses' in self.summary_labels:
            summary = tf.summary.scalar(name='loss-without-regularization', tensor=loss)
            self.summaries.append(summary)

        # Regularization losses
        losses = self.fn_regularization_losses(states=states, internals=internals)
        if len(losses) > 0:
            loss += tf.add_n(inputs=list(losses.values()))

        # Total loss summary
        if 'losses' in self.summary_labels or 'total-loss' in self.summary_labels:
            summary = tf.summary.scalar(name='total-loss', tensor=loss)
            self.summaries.append(summary)

        return loss

    def get_optimizer_kwargs(self, states, internals, actions, terminal, reward):
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

        Returns:
            Loss tensor of the size of the batch.
        """
        kwargs = dict()
        kwargs['time'] = self.timestep
        kwargs['variables'] = self.get_variables()
        kwargs['fn_loss'] = (
            lambda: self.fn_loss(states=states, internals=internals, actions=actions, terminal=terminal, reward=reward)
        )
        if self.distributed:  # not self.global_model?
            kwargs['global_variables'] = self.global_model.get_variables()
        return kwargs

    def tf_optimization(self, states, internals, actions, terminal, reward):
        """
        Creates the TensorFlow operations for performing an optimization update step based
        on the given input states and actions batch.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.

        Returns:
            The optimization operation.
        """
        optimizer_kwargs = self.get_optimizer_kwargs(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )
        return self.optimizer.minimize(**optimizer_kwargs)

                # if config.distributed and not config.global_model:


                # if config.distributed and not config.global_model:
                #     self.loss = tf.add_n(inputs=tf.losses.get_losses(scope=scope.name))
                #     local_grads_and_vars = self.optimizer.compute_gradients(loss=self.loss, var_list=self.variables)
                #     local_gradients = [grad for grad, var in local_grads_and_vars]
                #     global_gradients = list(zip(local_gradients, self.global_model.variables))
                #     self.update_local = tf.group(*(v1.assign(v2) for v1, v2 in zip(self.variables, self.global_model.variables)))
                #     self.optimize = tf.group(
                #         self.optimizer.apply_gradients(grads_and_vars=global_gradients),
                #         self.update_local,
                #         self.global_timestep.assign_add(tf.shape(self.reward)[0]))
                #     self.increment_global_episode = self.global_episode.assign_add(tf.count_nonzero(input_tensor=self.terminal, dtype=tf.int32))

    def tf_summarization(self):
        last_summary = tf.get_variable(name='last-summary', dtype=tf.int32, initializer=(-self.summary_frequency), trainable=False)

        def summarize():
            last_summary_updated = last_summary.assign(value=self.timestep)
            with tf.control_dependencies(control_inputs=(last_summary_updated,)):
                return tf.summary.merge(inputs=self.get_summaries())

        do_summarize = (self.timestep - last_summary >= self.summary_frequency)
        return tf.cond(pred=do_summarize, true_fn=summarize, false_fn=(lambda: ''))

    def create_output_operations(self, states, internals, actions, terminal, reward, deterministic):
        """
        Calls all the relevant TensorFlow functions for this model and hence creates all the
        TensorFlow operations involved.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            deterministic: If true, the action is chosen deterministically.
        """

        # Tensor fetched for model.act()
        increment_timestep = self.timestep.assign_add(delta=tf.shape(input=next(iter(states.values())))[0])  # Batch size
        with tf.control_dependencies(control_inputs=(increment_timestep,)):
            self.actions_and_internals = self.fn_actions_and_internals(
                states=states,
                internals=internals,
                deterministic=deterministic
            )

        # Tensor(s) fetched for model.update()
        self.loss_per_instance = self.fn_loss_per_instance(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )
        self.optimization = self.fn_optimization(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )
        self.summarization = self.fn_summarization()

    def get_variables(self):
        """
        Returns the TensorFlow variables used by the network.

        Returns:
            List of network variables.
        """
        return [self.variables[key] for key in sorted(self.variables)]

    def get_summaries(self):
        """
        Returns the TensorFlow summaries reported by the model

        Returns:
            List of summaries
        """
        return self.summaries

    def reset(self):
        """
        Resets the model to its initial state.

        Returns:
            A list containing the internal states initializations.
        """
        return list(self.internal_inits)

    def act(self, states, internals, deterministic=False):
        feed_dict = {state_input: (states[name],) for name, state_input in self.state_inputs.items()}
        feed_dict.update({internal_input: (internals[n],) for n, internal_input in enumerate(self.internal_inputs)})
        feed_dict[self.deterministic] = deterministic

        actions, internals = self.session.run(fetches=self.actions_and_internals, feed_dict=feed_dict)

        actions = {name: action[0] for name, action in actions.items()}
        internals = [internal[0] for internal in internals]
        return actions, internals

    def update(self, batch, return_loss_per_instance=False):
        """Generic batch update operation for Q-learning and policy gradient algorithms.

        Args:
            batch: Batch of experiences.

        Returns:

        """
        fetches = [self.optimization]

        # Optionally fetch loss per instance
        if return_loss_per_instance:
            fetches.append(self.loss_per_instance)

        # Periodically fetch summaries
        if len(self.summary_labels) > 0:
            fetches.append(self.summarization)

        feed_dict = self.update_feed_dict(batch=batch)

        # if self.distributed:
        #     fetches.extend(self.increment_global_episode for terminal in batch['terminals'] if terminal)

        fetched = self.session.run(fetches=fetches, feed_dict=feed_dict)

        if self.summary_writer is not None and len(self.summary_labels) > 0 and fetched[-1] != b'':
            self.summary_writer.add_summary(summary=fetched[-1], global_step=self.timestep)  # TODO: global_step?

        if return_loss_per_instance:
            return fetched[1]

    def update_feed_dict(self, batch):
        feed_dict = {state_input: batch['states'][name] for name, state_input in self.state_inputs.items()}
        feed_dict.update(
            {internal_input: batch['internals'][n]
                for n, internal_input in enumerate(self.internal_inputs)}
        )
        feed_dict.update(
            {action_input: batch['actions'][name]
                for name, action_input in self.action_inputs.items()}
        )
        feed_dict[self.terminal_input] = batch['terminal']
        feed_dict[self.reward_input] = batch['reward']
        return feed_dict

    def load_model(self, path):
        """
        Import model from path using tf.train.Saver.

        Args:
            path: Path to checkpoint

        Returns:

        """
        self.saver.restore(sess=self.session, save_path=path)

    def save_model(self, path, use_global_step=True):
        """
        Export model using a tf.train.Saver. Optionally append current time step as to not
        overwrite previous checkpoint file. Set to 'false' to be able to load model
        from exact path it was saved to in case of restarting program.

        Args:
            path: Model export directory
            use_global_step: Whether to append the current timestep to the checkpoint path.

        Returns:

        """
        if use_global_step:
            self.saver.save(sess=self.session, save_path=path, global_step=self.timestep)  # TODO: global_step?
        else:
            self.saver.save(sess=self.session, save_path=path)

        if self.summary_writer is not None:
            self.summary_writer.flush()
