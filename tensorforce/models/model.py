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
Models provide the general interface to TensorFlow functionality,
manages TensorFlow session and execution. In particular, a agent for reinforcement learning
always needs to provide a function that gives an action, and one to trigger updates.
A agent may use one more multiple neural networks and implement the update logic of a particular RL algorithm.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
import tensorflow as tf

from tensorforce import TensorForceError, util
from tensorforce.core.optimizers import Optimizer


log_levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'critical': logging.CRITICAL,
    'warning': logging.WARNING,
    'fatal': logging.FATAL
}


class Model(object):
    """
    Base model class.

    Each model requires the following configuration parameters:

    * `discount`: float of discount factor (gamma).
    * `learning_rate`: float of learning rate (alpha).
    * `optimizer`: string of optimizer to use (e.g. 'adam').
    * `device`: string of tensorflow device name.
    * `tf_saver`: boolean whether to save model parameters.
    * `tf_summary`: boolean indicating whether to use tensorflow summary file writer.
    * `log_level`: string containing logleve (e.g. 'info').
    * `distributed`: boolean indicating whether to use distributed tensorflow.
    * `global_model`: global model.
    * `session`: session to use.

    """
    allows_discrete_actions = None
    allows_continuous_actions = None

    default_config = dict(
        discount=0.97,
        learning_rate=0.0001,
        optimizer='adam',
        device=None,
        tf_saver=False,
        tf_summary=None,
        log_level='info',
        distributed=False,
        global_model=False,
        session=None
    )

    def __init__(self, config):
        """
        Creates a base reinforcement learning model with the specified configuration. Manages the creation
        of TensorFlow operations and provides a generic update method.
        
        Args:
            config: 
        """
        assert self.__class__.allows_discrete_actions is not None and self.__class__.allows_continuous_actions is not None
        config.default(Model.default_config)

        self.discount = config.discount
        self.distributed = config.distributed
        self.session = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_levels[config.log_level])

        if not config.distributed:
            assert not config.global_model and config.session is None
            tf.reset_default_graph()

        if config.distributed and not config.global_model:
            # Global and local model for asynchronous updates
            global_config = config.copy()
            global_config.optimizer = None
            global_config.global_model = True
            global_config.device = tf.train.replica_device_setter(1, worker_device=config.device, cluster=config.cluster_spec)
            self.global_model = self.__class__(config=global_config)
            self.global_timestep = self.global_model.timestep
            self.global_episode = self.global_model.episode
            self.global_variables = self.global_model.variables

        self.optimizer_args = None
        with tf.device(config.device):
            if config.distributed:
                if config.global_model:
                    self.timestep = tf.get_variable(name='timestep', dtype=tf.int32, initializer=0, trainable=False)
                    self.episode = tf.get_variable(name='episode', dtype=tf.int32, initializer=0, trainable=False)
                    scope_context = tf.variable_scope('global')
                else:
                    scope_context = tf.variable_scope('local')
                scope = scope_context.__enter__()

            self.create_tf_operations(config)

            if config.distributed:
                self.variables = tf.contrib.framework.get_variables(scope=scope)

            assert self.optimizer or (not config.distributed or config.global_model)
            if self.optimizer:
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
                else:
                    self.loss = tf.losses.get_total_loss()
                    if self.optimizer_args is not None:
                        self.optimizer_args['loss'] = self.loss
                        self.optimize = self.optimizer.minimize(self.optimizer_args)
                    else:
                        self.optimize = self.optimizer.minimize(self.loss)

            if config.distributed:
                scope_context.__exit__(None, None, None)

        self.saver = tf.train.Saver()
        if config.tf_summary is not None:
            self.writer = tf.summary.FileWriter(config.tf_summary, graph=tf.get_default_graph())
        else:
            self.writer = None

        if not config.distributed:
            self.set_session(tf.Session())
            self.session.run(tf.global_variables_initializer())
            tf.get_default_graph().finalize()

    def create_tf_operations(self, config):
        """
        Creates generic TensorFlow operations and placeholders required for models.
        
        Args:
            config: Model configuration which must contain entries for states and actions.

        Returns:

        """
        self.action_taken = dict()
        self.internal_inputs = list()
        self.internal_outputs = list()
        self.internal_inits = list()

        # Placeholders
        with tf.variable_scope('placeholder'):
            # States
            self.state = dict()
            for name, state in config.states.items():
                self.state[name] = tf.placeholder(dtype=util.tf_dtype(state.type), shape=(None,) + tuple(state.shape), name=name)

            # Actions
            self.action = dict()
            self.discrete_actions = []
            self.continuous_actions = []
            for name, action in config.actions:
                if action.continuous:
                    if not self.__class__.allows_continuous_actions:
                        raise TensorForceError("Error: Model does not support continuous actions.")
                    self.action[name] = tf.placeholder(dtype=util.tf_dtype('float'), shape=(None,) + tuple(action.shape), name=name)
                else:
                    if not self.__class__.allows_discrete_actions:
                        raise TensorForceError("Error: Model does not support discrete actions.")
                    self.action[name] = tf.placeholder(dtype=util.tf_dtype('int'), shape=(None,) + tuple(action.shape), name=name)

            # Reward & terminal
            self.reward = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')
            self.terminal = tf.placeholder(dtype=tf.bool, shape=(None,), name='terminal')

            # Deterministic action flag
            self.deterministic = tf.placeholder(dtype=tf.bool, shape=(), name='deterministic')

        # Optimizer
        if config.optimizer is not None:
            with tf.variable_scope('optimization'):
                self.optimizer = Optimizer.from_config(
                    config=config.optimizer,
                    kwargs=dict(learning_rate=config.learning_rate)
                )
        else:
            self.optimizer = None

    def set_session(self, session):
        assert self.session is None
        self.session = session

    def reset(self):
        """
        Resets the internal state to the initial state.
        Returns: A list containing the internal_inits field.

        """
        return list(self.internal_inits)

    def get_action(self, state, internal, deterministic=False):
        fetches = {action: action_taken for action, action_taken in self.action_taken.items()}
        fetches.update({n: internal_output for n, internal_output in enumerate(self.internal_outputs)})

        feed_dict = {state_input: (state[name],) for name, state_input in self.state.items()}
        feed_dict.update({internal_input: (internal[n],) for n, internal_input in enumerate(self.internal_inputs)})
        feed_dict[self.deterministic] = deterministic

        fetched = self.session.run(fetches=fetches, feed_dict=feed_dict)

        action = {name: fetched[name][0] for name in self.action}
        internal = [fetched[n][0] for n in range(len(self.internal_outputs))]
        return action, internal

    def update(self, batch):
        """Generic batch update operation for Q-learning and policy gradient algorithms.
         Takes a batch of experiences,
         
        Args:
            batch: Batch of experiences. 

        Returns:

        """
        if self.optimizer is None:
            return

        fetches = [self.optimize, self.loss, self.loss_per_instance]

        feed_dict = {state: batch['states'][name] for name, state in self.state.items()}
        feed_dict.update({action: batch['actions'][name] for name, action in self.action.items()})
        feed_dict[self.reward] = batch['rewards']
        feed_dict[self.terminal] = batch['terminals']
        feed_dict.update({internal: batch['internals'][n] for n, internal in enumerate(self.internal_inputs)})

        if self.distributed:
            fetches.extend(self.increment_global_episode for terminal in batch['terminals'] if terminal)
            loss, loss_per_instance = self.session.run(fetches=fetches, feed_dict=feed_dict)[1:3]
        else:
            loss, loss_per_instance = self.session.run(fetches=fetches, feed_dict=feed_dict)[1:]

        self.logger.debug('Computed update with loss = {}.'.format(loss))

        return loss, loss_per_instance

    def load_model(self, path):
        self.saver.restore(self.session, path)

    def save_model(self, path):
        self.saver.save(self.session, path)
