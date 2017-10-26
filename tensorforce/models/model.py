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
from tensorforce.core.optimizers import Optimizer, GlobalOptimizer


class Model(object):
    """
    Base class for all (TensorFlow-based) models.
    """

    def __init__(self, states_spec, actions_spec, config, **kwargs):

        # States and actions specifications
        self.states_spec = states_spec
        self.actions_spec = actions_spec

        # Discount factor
        self.discount = config.discount

        # Reward normalization
        assert isinstance(config.normalize_rewards, bool)
        self.normalize_rewards = config.normalize_rewards

        # Variable noise
        assert config.variable_noise is None or config.variable_noise > 0.0
        self.variable_noise = config.variable_noise

        # TensorFlow summaries
        self.summary_labels = set(config.summary_labels or ())

        # Variables and summaries
        self.variables = dict()
        self.all_variables = dict()
        self.summaries = list()

        if not config.local_model or not config.replica_model:
            # If not local_model mode or not internal global model
            self.default_graph = tf.Graph().as_default()
            self.graph = self.default_graph.__enter__()

        if config.cluster_spec is None:
            if config.parameter_server or config.replica_model or config.local_model:
                raise TensorForceError("Invalid config value for distributed mode.")
            self.device = config.device
            self.global_model = None

        elif config.parameter_server:
            if config.replica_model or config.local_model:
                raise TensorForceError("Invalid config value for distributed mode.")
            self.device = config.device
            self.global_model = None

        elif config.replica_model:
            self.device = tf.train.replica_device_setter(worker_device=config.device, cluster=config.cluster_spec)
            self.global_model = None

        elif config.local_model:
            if config.replica_model:
                raise TensorForceError("Invalid config value for distributed mode.")
            self.device = config.device

            global_config = config.copy()
            global_config.set(key='replica_model', value=True)

            self.global_model = self.__class__(
                states_spec=states_spec,
                actions_spec=actions_spec,
                config=global_config,
                **kwargs
            )

        else:
            raise TensorForceError("Invalid config value for distributed mode.")

        with tf.device(device_name_or_function=self.device):

            # Timestep and episode
            # TODO: various modes !!!
            if self.global_model is None:
                # TODO: Variables seem to re-initialize in the beginning every time a runner starts
                self.timestep = tf.get_variable(name='timestep', dtype=tf.int32, initializer=0, trainable=False)
                self.episode = tf.get_variable(name='episode', dtype=tf.int32, initializer=0, trainable=False)
            else:
                self.timestep = self.global_model.timestep
                self.episode = self.global_model.episode

            with tf.name_scope(name=config.scope):

                def custom_getter(getter, name, registered=False, **kwargs):
                    variable = getter(name=name, **kwargs)  # Top-level, hence no 'registered'
                    if not registered and not name.startswith('optimization'):
                        self.all_variables[name] = variable
                        if kwargs.get('trainable', True):
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
                if config.optimizer is None:
                    self.optimizer = None
                elif config.local_model and not config.replica_model:
                    # If local_model mode and not internal global model
                    self.optimizer = GlobalOptimizer(optimizer=config.optimizer)
                else:
                    self.optimizer = Optimizer.from_spec(spec=config.optimizer)

                # Create output fetch operations
                self.create_output_operations(
                    states=states,
                    internals=internals,
                    actions=actions,
                    terminal=terminal,
                    reward=reward,
                    deterministic=self.deterministic
                )

        if config.local_model and config.replica_model:
            # If local_model mode and internal global model
            return

        # Local and global initialize operations
        if config.local_model:
            init_op = tf.variables_initializer(
                var_list=(self.global_model.get_variables(include_non_trainable=True))
            )
            local_init_op = tf.variables_initializer(
                var_list=(self.get_variables(include_non_trainable=True))
            )

        else:
            init_op = tf.variables_initializer(
                var_list=(self.get_variables(include_non_trainable=True))
            )
            local_init_op = None

        # Summary operation
        if len(self.get_summaries()) > 0:
            summary_op = tf.summary.merge(inputs=self.get_summaries())
        else:
            summary_op = None

        # TODO: MonitoredSession or so?
        self.supervisor = tf.train.Supervisor(
            is_chief=(config.task_index == 0),
            init_op=init_op,
            local_init_op=local_init_op,
            logdir=config.model_directory,
            summary_op=summary_op,
            global_step=self.timestep,
            save_summaries_secs=config.summary_frequency,
            save_model_secs=config.save_frequency
            # checkpoint_basename='model.ckpt'
            # session_manager=None
        )

        # tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:{}/cpu:0'.format(self.task_index)])
        if config.parameter_server:
            self.server = tf.train.Server(
                server_or_cluster_def=config.cluster_spec,
                job_name='ps',
                task_index=config.task_index,
                # config=tf.ConfigProto(device_filters=["/job:ps"])
                # config=tf.ConfigProto(
                #     inter_op_parallelism_threads=2,
                #     log_device_placement=True
                # )
            )

            # Param server does nothing actively
            self.server.join()

        elif config.cluster_spec is not None:
            self.server = tf.train.Server(
                server_or_cluster_def=config.cluster_spec,
                job_name='worker',
                task_index=config.task_index,
                # config=tf.ConfigProto(device_filters=["/job:ps"])
                # config=tf.ConfigProto(
                #     inter_op_parallelism_threads=2,
                #     log_device_placement=True
                # )
            )

            self.managed_session = self.supervisor.managed_session(
                master=self.server.target,
                start_standard_services=True
            )
            self.session = self.managed_session.__enter__()

        else:
            self.managed_session = self.supervisor.managed_session(
                start_standard_services=True
            )
            self.session = self.managed_session.__enter__()

    def close(self):
        self.managed_session.__exit__(None, None, None)
        self.supervisor.stop()
        self.default_graph.__exit__(None, None, None)

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

        # TensorFlow functions
        self.fn_discounted_cumulative_reward = tf.make_template(
            name_='discounted-cumulative-reward',
            func_=self.tf_discounted_cumulative_reward,
            custom_getter_=custom_getter
        )
        self.fn_actions_and_internals = tf.make_template(
            name_='actions-and-internals',
            func_=self.tf_actions_and_internals,
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
        # self.fn_summarization = tf.make_template(
        #     name_='summarization',
        #     func_=self.tf_summarization,
        #     custom_getter_=custom_getter
        # )

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

        def cumulate(cumulative, reward_and_terminal):
            rew, term = reward_and_terminal
            return tf.where(
                condition=term,
                x=rew,
                y=(rew + cumulative * discount)
            )

        # Reverse since reward cumulation is calculated right-to-left, but tf.scan only works left-to-right
        reward = tf.reverse(tensor=reward, axis=(0,))
        terminal = tf.reverse(tensor=terminal, axis=(0,))

        reward = tf.scan(fn=cumulate, elems=(reward, terminal), initializer=final_reward)

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
        if self.global_model is not None:
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
        if self.optimizer is None:
            return tf.no_op()
        else:
            optimizer_kwargs = self.get_optimizer_kwargs(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward
            )
            return self.optimizer.minimize(**optimizer_kwargs)

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

        # Create graph by calling the functions corresponding to model.act() / model.update(), to initialize variables.
        # TODO: Could call reset here, but would have to move other methods below reset.
        self.fn_actions_and_internals(
            states=states,
            internals=internals,
            deterministic=deterministic
        )
        self.fn_optimization(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )

        # Tensor fetched for model.act()
        operations = list()
        if self.variable_noise is not None and self.variable_noise > 0.0:
            # Add variable noise
            noise_deltas = list()
            for variable in self.get_variables():
                noise_delta = tf.random_normal(shape=util.shape(variable), mean=0.0, stddev=self.variable_noise)
                noise_deltas.append(noise_delta)
                operations.append(variable.assign_add(delta=noise_delta))

        # Retrieve actions and internals
        with tf.control_dependencies(control_inputs=operations):
            self.actions_internals_timestep = self.fn_actions_and_internals(
                states=states,
                internals=internals,
                deterministic=deterministic
            )

        # Increment timestep
        increment_timestep = tf.shape(input=next(iter(states.values())))[0]
        increment_timestep = self.timestep.assign_add(delta=increment_timestep)
        operations = [increment_timestep]

        # Subtract variable noise
        if self.variable_noise is not None and self.variable_noise > 0.0:
            for variable, noise_delta in zip(self.get_variables(), noise_deltas):
                operations.append(variable.assign_sub(delta=noise_delta))

        with tf.control_dependencies(control_inputs=operations):
            # Trivial operation to enforce control dependency
            self.actions_internals_timestep += (self.timestep + 0,)

        # Tensor fetched for model.observe()
        increment_episode = tf.count_nonzero(input_tensor=terminal, dtype=tf.int32)
        increment_episode = self.episode.assign_add(delta=increment_episode)
        # TODO: add up rewards per episode and add summary_label 'episode-reward'
        with tf.control_dependencies(control_inputs=(increment_episode,)):
            self.episode_increment = tf.no_op()

        # Tensor(s) fetched for model.update()
        self.optimization = self.fn_optimization(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )
        self.loss_per_instance = self.fn_loss_per_instance(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )

    def get_variables(self, include_non_trainable=False):
        """
        Returns the TensorFlow variables used by the model.

        Returns:
            List of variables.
        """

        if include_non_trainable:
                # optimizer variables and timestep/episode only included if 'include_non_trainable' set
            model_variables = [self.all_variables[key] for key in sorted(self.all_variables)]

            if self.optimizer is None:
                return model_variables + [self.timestep, self.episode]

            else:
                optimizer_variables = self.optimizer.get_variables()
                return model_variables + optimizer_variables + [self.timestep, self.episode]

        else:
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
        name = next(iter(self.states_spec))
        batched = (states[name].ndim != len(self.states_spec[name]['shape']))

        fetches = list(self.actions_internals_timestep)

        if batched:
            feed_dict = {state_input: states[name] for name, state_input in self.state_inputs.items()}
            feed_dict.update({internal_input: internals[n] for n, internal_input in enumerate(self.internal_inputs)})
        else:
            feed_dict = {state_input: (states[name],) for name, state_input in self.state_inputs.items()}
            feed_dict.update({internal_input: (internals[n],) for n, internal_input in enumerate(self.internal_inputs)})

        feed_dict[self.deterministic] = deterministic

        actions, internals, timestep = self.session.run(fetches=fetches, feed_dict=feed_dict)

        if not batched:
            actions = {name: action[0] for name, action in actions.items()}
            internals = [internal[0] for internal in internals]

        return actions, internals, timestep

    def observe(self, terminal, reward, batched=False):
        fetches = [self.episode_increment, self.episode]

        if batched:
            feed_dict = {self.terminal_input: terminal, self.reward_input: reward}
        else:
            feed_dict = {self.terminal_input: (terminal,), self.reward_input: (reward,)}

        _, episode = self.session.run(fetches=fetches, feed_dict=feed_dict)

        return episode

    def update(self, batch, return_loss_per_instance=False):
        fetches = [self.optimization]

        # Optionally fetch loss per instance
        if return_loss_per_instance:
            fetches.append(self.loss_per_instance)

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

        # if self.distributed:
        #     fetches.extend(self.increment_global_episode for terminal in batch['terminals'] if terminal)

        fetched = self.session.run(fetches=fetches, feed_dict=feed_dict)

        if return_loss_per_instance:
            return fetched[1]

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
