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

import logging
import tensorflow as tf

from tensorforce import TensorForceError, util
from tensorforce.core.optimizers import Optimizer


class Model(object):
    """
    Base class for all models
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
        """
        Creates a base reinforcement learning model with the specified configuration. Manages the creation
        of TensorFlow operations and provides a generic update method.

        Args:
            config:
        """

        self.states_spec = states_spec
        self.actions_spec = actions_spec

        self.discount = config.discount
        # self.distributed = config.distributed
        self.session = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(util.log_levels[config.log_level])

        # if not config.distributed:
        #     assert not config.global_model and config.session is None
        tf.reset_default_graph()

        # if config.distributed and not config.global_model:
        #     # Global and local model for asynchronous updates
        #     global_config = config.copy()
        #     global_config.optimizer = None
        #     global_config.global_model = True
        #     global_config.device = tf.train.replica_device_setter(1, worker_device=config.device, cluster=config.cluster_spec)
        #     self.global_model = self.__class__(config=global_config)
        #     self.global_timestep = self.global_model.global_timestep
        #     self.global_episode = self.global_model.episode
        #     self.global_variables = self.global_model.variables

        with tf.device(device_name_or_function=None):  # config.device
            # if config.distributed:
            #     if config.global_model:
            #         self.global_timestep = tf.get_variable(name='timestep', dtype=tf.int32, initializer=0, trainable=False)
            #         self.episode = tf.get_variable(name='episode', dtype=tf.int32, initializer=0, trainable=False)
            #         scope_context = tf.variable_scope('global')
            #     else:
            #         scope_context = tf.variable_scope('local')
            #     scope = scope_context.__enter__()

            # DOESN'T WORK
            # if config.distributed:
            #     self.variables = tf.contrib.framework.get_variables(scope=scope)

            # Optimizer
            # if config.optimizer is None:
            #     assert not config.distributed or config.global_model
            #     self.optimizer = None
            # else:

            # if self.optimizer is not None:
            #     if config.distributed and not config.global_model:
            #         self.loss = tf.add_n(inputs=tf.losses.get_losses(scope=scope.name))
            #         local_grads_and_vars = self.optimizer.compute_gradients(loss=self.loss, var_list=self.variables)
            #         local_gradients = [grad for grad, var in local_grads_and_vars]
            #         global_gradients = list(zip(local_gradients, self.global_model.variables))
            #         self.update_local = tf.group(*(v1.assign(v2) for v1, v2 in zip(self.variables, self.global_model.variables)))
            #         self.optimize = tf.group(
            #             self.optimizer.apply_gradients(grads_and_vars=global_gradients),
            #             self.update_local,
            #             self.global_timestep.assign_add(tf.shape(self.reward)[0]))
            #         self.increment_global_episode = self.global_episode.assign_add(tf.count_nonzero(input_tensor=self.terminal, dtype=tf.int32))

            self.variables = dict()

            with tf.name_scope(name=config.scope):
                def custom_getter(getter, name, *args, **kwargs):
                    variable = getter(name=name, *args, **kwargs)
                    if not name.startswith('optimization'):
                        self.variables[name] = variable
                    return variable

                # Create placeholders, tf functions, internals, etc
                self.initialize(custom_getter=custom_getter)

                # Input tensors
                states = self.get_states(states=self.state_inputs)
                actions = self.get_actions(actions=self.action_inputs)
                terminal = tf.identity(input=self.terminal_input)
                reward = self.get_reward(states=states, terminal=terminal, reward=self.reward_input, internals=self.internal_inputs)

                # Stop gradients for input preprocessing
                states = {name: tf.stop_gradient(input=state) for name, state in states.items()}
                actions = {name: tf.stop_gradient(input=action) for name, action in actions.items()}
                reward = tf.stop_gradient(input=reward)

                # Optimizer
                self.optimizer = Optimizer.from_spec(spec=config.optimizer)

                # Tensor fetched for model.act()
                increment_time = self.time.assign_add(delta=1)
                with tf.control_dependencies(control_inputs=(increment_time,)):
                    self.actions_and_internals = self.fn_actions_and_internals(states=states, internals=self.internal_inputs, deterministic=self.deterministic)

                # Tensor(s) fetched for model.update()
                self.loss_per_instance = self.fn_loss_per_instance(states=states, actions=actions, reward=reward, terminal=terminal, internals=self.internal_inputs)
                self.optimization = self.fn_optimization(states=states, actions=actions, reward=reward, terminal=terminal, internals=self.internal_inputs)

                # if config.distributed:
                #     scope_context.__exit__(None, None, None)

        self.saver = tf.train.Saver()

        if config.tf_summary is not None:
            self.summary_level = config.tf_summary['level']  # summary level not used !!!
            self.summary_interval = config.tf_summary['interval']
            self.last_summary_step = -config.tf_summary['interval']

            # create a summary for total loss
            tf.summary.scalar('total-loss', self.loss)

            # create summary writer
            self.writer = tf.summary.FileWriter(logdir=config.tf_summary['logdir'], graph=tf.get_default_graph())

            # create summaries based on summary level
            if self.summary_level >= 2:  # trainable variables
                for v in tf.trainable_variables():
                    tf.summary.histogram(v.name, v)

            # merge all summaries
            self.tf_summaries = tf.summary.merge_all()

            # create a separate summary for episode rewards
            self.tf_episode_reward = tf.placeholder(tf.float32, name='episode-reward-placeholder')
            self.episode_reward_summary = tf.summary.scalar('episode-reward', self.tf_episode_reward)
        else:
            self.writer = None
            self.summary_level = None

        self.timestep = 0

        # if not config.distributed:
        self.set_session(tf.Session())
        self.session.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()

    def initialize(self, custom_getter):
        # States
        self.state_inputs = dict()
        for name, state in self.states_spec.items():
            self.state_inputs[name] = tf.placeholder(dtype=util.tf_dtype(state['type']), shape=(None,) + tuple(state['shape']), name=name)

        # Actions
        self.action_inputs = dict()
        for name, action in self.actions_spec.items():
            self.action_inputs[name] = tf.placeholder(dtype=util.tf_dtype(action['type']), shape=(None,) + tuple(action['shape']), name=name)

        # Terminal
        self.terminal_input = tf.placeholder(dtype=tf.bool, shape=(None,), name='terminal')

        # Reward
        self.reward_input = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')

        # Internal states
        self.internal_inputs = list()
        self.internal_inits = list()

        # Deterministic action flag
        self.deterministic = tf.placeholder(dtype=tf.bool, shape=(), name='deterministic')

        # Time
        # various modes !!!
        self.time = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, expected_shape=())

        # TensorFlow functions
        self.fn_discounted_cumulative_reward = tf.make_template(
            name_=('discounted_cumulative_reward'),
            func_=self.tf_discounted_cumulative_reward,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )
        self.fn_actions_and_internals = tf.make_template(
            name_='actions_and_internals',
            func_=self.tf_actions_and_internals,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )
        self.fn_loss_per_instance = tf.make_template(
            name_='loss_per_instance',
            func_=self.tf_loss_per_instance,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )
        self.fn_regularization_losses = tf.make_template(
            name_='regularization_losses',
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

    def get_states(self, states):
        # preprocessing could go here
        return {name: tf.identity(input=state) for name, state in states.items()}

    def get_actions(self, actions):
        # preprocessing could go here
        return {name: tf.identity(input=action) for name, action in actions.items()}

    def get_reward(self, states, terminal, reward, internals):
        # preprocessing could go here
        return tf.identity(input=reward)

    def tf_discounted_cumulative_reward(self, reward, terminal, discount, final_reward=0.0):
        # TODO: n-step cumulative reward (particularly for envs without terminal)

        def fn_scan(cumulative, reward_and_terminal):
            reward_, terminal_ = reward_and_terminal
            return tf.cond(
                pred=terminal_,
                true_fn=(lambda: reward_),
                false_fn=(lambda: reward_ + cumulative * discount)
            )

        reward = tf.reverse(tensor=reward, axis=(0,))
        terminal = tf.reverse(tensor=terminal, axis=(0,))
        reward = tf.scan(fn=fn_scan, elems=(reward, terminal), initializer=final_reward)
        return tf.reverse(tensor=reward, axis=(0,))

    # def update_inputs(self, ...???):
    #     # for batching ??? for sequence format?
    #     return self.states, self.actions, self.terminal, self.reward, self.internals

    def tf_actions_and_internals(self, states):
        raise NotImplementedError

    def tf_loss_per_instance(self, states, actions, terminal, reward, internals):
        raise NotImplementedError

    def tf_regularization_losses(self, states, internals):
        return dict()

    def tf_loss(self, states, actions, terminal, reward, internals):
        loss_per_instance = self.fn_loss_per_instance(states=states, actions=actions, terminal=terminal, reward=reward, internals=internals)

        loss = tf.reduce_mean(input_tensor=loss_per_instance, axis=0)

        losses = self.fn_regularization_losses(states=states, internals=internals)
        if len(losses) > 0:
            loss += tf.add_n(inputs=list(losses.values()))

        return loss

    def get_optimizer_kwargs(self, states, actions, terminal, reward, internals):
        kwargs = dict()
        kwargs['variables'] = self.get_variables()
        kwargs['time'] = self.time
        kwargs['fn_loss'] = (lambda: self.fn_loss(states=states, actions=actions, terminal=terminal, reward=reward, internals=internals))
        return kwargs

    def tf_optimization(self, states, actions, terminal, reward, internals):
        optimizer_kwargs = self.get_optimizer_kwargs(states=states, actions=actions, terminal=terminal, reward=reward, internals=internals)
        return self.optimizer.minimize(**optimizer_kwargs)

    def get_variables(self):
        return [self.variables[key] for key in sorted(self.variables)]

    def set_session(self, session):
        assert self.session is None
        self.session = session

    def reset(self):
        """
        Resets the internal state to the initial state.
        Returns: A list containing the internal_inits field.

        """
        return list(self.internal_inits)

    def act(self, states, internals, deterministic=False):
        self.timestep += 1

        feed_dict = {state_input: (states[name],) for name, state_input in self.state_inputs.items()}
        feed_dict.update({internal_input: (internals[n],) for n, internal_input in enumerate(self.internal_inputs)})
        feed_dict[self.deterministic] = deterministic

        actions, internals = self.session.run(fetches=self.actions_and_internals, feed_dict=feed_dict)

        actions = {name: action[0] for name, action in actions.items()}
        internals = [internal[0] for internal in internals]
        return actions, internals

    def update(self, batch, return_loss_per_instance=False):
        """Generic batch update operation for Q-learning and policy gradient algorithms.
         Takes a batch of experiences,

        Args:
            batch: Batch of experiences.

        Returns:

        """
        fetches = self.optimization
        if return_loss_per_instance:
            fetches = (fetches, self.loss_per_instance)

        feed_dict = self.update_feed_dict(batch=batch)

        # check if we should write summaries
        write_summaries = self.should_write_summaries(self.timestep)
        if write_summaries:
            self.last_summary_step = self.timestep
            fetches.append(self.tf_summaries)

        # if self.distributed:
        #     fetches.extend(self.increment_global_episode for terminal in batch['terminals'] if terminal)

        fetched = self.session.run(fetches=fetches, feed_dict=feed_dict)

        # loss, loss_per_instance = returns[1:3]
        # if write_summaries:
        #     self.write_summaries(returns[3])

        # self.logger.debug('Computed update with loss = {}.'.format(loss))

        if return_loss_per_instance:
            return fetched[1]

    def update_feed_dict(self, batch):
        feed_dict = {state_input: batch['states'][name] for name, state_input in self.state_inputs.items()}
        feed_dict.update({action_input: batch['actions'][name] for name, action_input in self.action_inputs.items()})
        feed_dict[self.terminal_input] = batch['terminal']
        feed_dict[self.reward_input] = batch['reward']
        feed_dict.update({internal_input: batch['internals'][n] for n, internal_input in enumerate(self.internal_inputs)})
        return feed_dict

    def load_model(self, path):
        """
        Import model from path using tf.train.Saver.

        Args:
            path: Path to checkpoint

        Returns:

        """
        self.saver.restore(self.session, path)

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
            self.saver.save(self.session, path, global_step=self.timestep)
        else:
            self.saver.save(self.session, path)

    def should_write_summaries(self, num_updates):
        return self.writer is not None and self.timestep >= self.last_summary_step + self.summary_interval

    def write_summaries(self, summaries):
        self.writer.add_summary(summaries, global_step=self.timestep)

    def write_episode_reward_summary(self, episode_reward):
        if self.writer is not None:
            reward_summary = self.session.run(self.episode_reward_summary, feed_dict={self.tf_episode_reward: episode_reward})
            self.writer.add_summary(reward_summary, global_step=self.timestep)
