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
A agent may use one more multiple neural networks and implement the update logic of a particular
RL algorithm.

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
import tensorflow as tf

from tensorforce import TensorForceError, util


log_levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'critical': logging.CRITICAL,
    'warning': logging.WARNING,
    'fatal': logging.FATAL
}


class Model(object):

    allows_discrete_actions = None
    allows_continuous_actions = None
    default_config = dict(
        discount=0.97,
        exploration=None,
        exploration_args=None,
        exploration_kwargs=None,
        learning_rate=0.0001,
        optimizer='adam',
        optimizer_args=None,
        optimizer_kwargs=None,
        device=None,
        logging=None,
        tf_saver=None,
        tf_summary=None
    )

    def __init__(self, config):
        """
        :param config: Configuration parameters
        """
        assert self.__class__.allows_discrete_actions is not None and self.__class__.allows_continuous_actions is not None
        config.default(Model.default_config)

        self.num_actions = config.actions
        self.continuous = config.continuous
        self.discount = config.discount

        # tf, initialization, loss, optimization
        tf.reset_default_graph()
        self.session = tf.Session()
        with tf.device(config.device):
            self.create_tf_operations(config)
            if self.optimizer:
                self.loss = tf.losses.get_total_loss()
                self.optimize = self.optimizer.minimize(self.loss)
        if config.tf_saver:
            self.saver = tf.train.Saver()
        else:
            self.saver = None
        if config.tf_summary:
            self.writer = tf.summary.FileWriter(config.tf_summary, graph=tf.get_default_graph())
        else:
            self.writer = None
        self.session.run(tf.global_variables_initializer())

        # logger
        if config.logging:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(log_levels[config.logging])
        else:
            self.logger = None

    def create_tf_operations(self, config):
        # placeholders
        with tf.variable_scope('placeholders'):

            # states
            self.state = dict()
            for name, state in config.states.items():
                self.state[name] = tf.placeholder(dtype=util.tf_dtype(state['type']), shape=(None,) + tuple(state['shape']), name=name)

            # actions
            self.action = dict()
            self.discrete_actions = []
            self.continuous_actions = []
            for name, action in config.actions.items():
                if action['continuous']:
                    if not self.__class__.allows_continuous_actions:
                        raise TensorForceError()
                    self.action[name] = tf.placeholder(dtype=util.tf_dtype('float'), shape=(None,), name=name)
                else:
                    if not self.__class__.allows_discrete_actions:
                        raise TensorForceError()
                    self.action[name] = tf.placeholder(dtype=util.tf_dtype('int'), shape=(None,), name=name)

            # reward
            self.reward = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')

        # optimizer
        if config.optimizer:
            learning_rate = config.learning_rate
            with tf.variable_scope('optimization'):
                if config.optimizer == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                else:
                    optimizer = util.module(config.optimizer)
                    args = config.optimizer_args or ()
                    kwargs = config.optimizer_kwargs or {}
                    self.optimizer = optimizer(learning_rate, *args, **kwargs)
        else:
            self.optimizer = None

    def reset(self):
        self.internals = None

    def get_action(self, state):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def load_model(self, path):
        self.saver.restore(self.session, path)

    def save_model(self, path):
        self.saver.save(self.session, path)
