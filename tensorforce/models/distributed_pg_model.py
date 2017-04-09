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
Model for the distributed advantage actor critic.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import logging

from tensorforce.config import create_config
from tensorforce.models.baselines import LinearValueFunction
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.models.policies import CategoricalOneHotPolicy
from tensorforce.models.policies import GaussianPolicy
from tensorforce.util.config_util import get_function
from tensorforce.util.experiment_util import global_seed
from tensorforce.util.exploration_util import exploration_mode
from tensorforce.util.math_util import zero_mean_unit_variance, discount


class DistributedPGModel(object):
    default_config = {}

    def __init__(self, config, scope, task_index, cluster_spec, define_network=None):
        """

        A distributed agent must synchronise local and global parameters under different
        scopes.

        :param config: Configuration parameters
        :param scope: TensorFlow scope
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.session = None
        self.saver = None
        self.config = create_config(config, default=self.default_config)
        self.scope = scope
        self.task_index = task_index
        self.batch_size = self.config.batch_size
        self.action_count = self.config.actions
        self.generalized_advantage_estimation = self.config.use_gae
        self.gae_lambda = self.config.gae_lambda

        self.gamma = self.config.gamma
        self.continuous = self.config.continuous
        self.normalize_advantage = self.config.normalise_advantage
        self.episode_length = tf.placeholder(tf.int32, (None,), name='episode_length')

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        if define_network is None:
            self.define_network = NeuralNetwork.layered_network(self.config.network_layers)
        else:
            self.define_network = define_network

        # This is the scope used to prefix variable creation for distributed TensorFlow
        self.batch_shape = [None]
        self.deterministic_mode = config.get('deterministic_mode', False)
        self.alpha = config.get('alpha', 0.001)
        self.optimizer = None

        self.worker_device = "/job:worker/task:{}/cpu:0".format(task_index)
        self.state_shape = tuple(self.config.state_shape)

        with tf.device(tf.train.replica_device_setter(1, worker_device=self.worker_device, cluster=cluster_spec)):
            with tf.variable_scope("global"):
                self.global_state = tf.placeholder(tf.float32, (None, None) + self.state_shape,
                                                   name="global_state")

                self.global_network = NeuralNetwork(self.define_network, [self.global_state], episode_length=self.episode_length)
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                self.global_states = self.global_network.internal_state_inits

                self.global_prev_action_means = tf.placeholder(tf.float32, (None, None, self.action_count), name='prev_actions')

                if self.continuous:
                    self.global_policy = GaussianPolicy(self.global_network, self.session, self.global_state, self.random,
                                                 self.action_count, 'gaussian_policy')
                    self.global_prev_action_log_stds = tf.placeholder(tf.float32, (None, None, self.action_count))

                    self.global_prev_dist = dict(policy_output=self.global_prev_action_means,
                                          policy_log_std=self.global_prev_action_log_stds)

                else:
                    self.global_policy = CategoricalOneHotPolicy(self.global_network, self.session, self.global_state, self.random,
                                                          self.action_count, 'categorical_policy')
                    self.global_prev_dist = dict(policy_output=self.global_prev_action_means)

                # Probability distribution used in the current policy
                self.global_baseline_value_function = LinearValueFunction()

            # self.optimizer = config.get('optimizer')
            # self.optimizer_args = config.get('optimizer_args', [])
            # self.optimizer_kwargs = config.get('optimizer_kwargs', {})

        exploration = config.get('exploration')
        if not exploration:
            self.exploration = exploration_mode['constant'](self, 0)
        else:
            args = config.get('exploration_args', [])
            kwargs = config.get('exploration_kwargs', {})
            self.exploration = exploration_mode[exploration](self, *args, **kwargs)

        self.create_training_operations()

    def set_session(self, session):
        self.session = session

        # Session in policy was still 'None' when
        # we initialised policy, hence need to set again
        self.policy.session = session

    def create_training_operations(self):
        """
        Currently a duplicate of the pg agent logic, to be made generic later to allow
        all models to be executed asynchronously/distributed seamlessly.

        """
        # TODO rewrite agent logic so core update logic can be composed into
        # TODO distributed logic

        with tf.device(self.worker_device):
            with tf.variable_scope("local"):
                self.state = tf.placeholder(tf.float32, (None, None) + self.state_shape,
                                            name="state")
                self.prev_action_means = tf.placeholder(tf.float32, (None, None, self.action_count), name='prev_actions')

                self.local_network = NeuralNetwork(self.define_network, [self.state], episode_length=self.episode_length)
                self.local_states = self.local_network.internal_state_inits

                # TODO possibly problematic, check
                self.local_step = self.global_step

                if self.continuous:
                    self.policy = GaussianPolicy(self.local_network, self.session, self.state, self.random,
                                                 self.action_count, 'gaussian_policy')
                    self.prev_action_log_stds = tf.placeholder(tf.float32, (None, None, self.action_count))

                    self.prev_dist = dict(policy_output=self.prev_action_means,
                                          policy_log_std=self.prev_action_log_stds)

                else:
                    self.policy = CategoricalOneHotPolicy(self.local_network, self.session, self.state, self.random,
                                                          self.action_count, 'categorical_policy')
                    self.prev_dist = dict(policy_output=self.prev_action_means)

                # Probability distribution used in the current policy
                self.baseline_value_function = LinearValueFunction()

            self.actions = tf.placeholder(tf.float32, (None, None, self.action_count), name='actions')
            self.advantage = tf.placeholder(tf.float32, shape=(None, None, 1), name='advantage')

            self.dist = self.policy.get_distribution()
            self.log_probabilities = self.dist.log_prob(self.policy.get_policy_variables(), self.actions)

            # Concise: Get log likelihood of actions, weigh by advantages, compute gradient on that
            self.loss = -tf.reduce_mean(self.log_probabilities * self.advantage, name="loss_op")

            self.gradients = tf.gradients(self.loss, self.local_network.variables)

            grad_var_list = list(zip(self.gradients, self.global_network.variables))

            global_step_inc = self.global_step.assign_add(tf.shape(self.state)[0])

            self.assign_global_to_local = tf.group(*[v1.assign(v2) for v1, v2 in
                                                     zip(self.local_network.variables,
                                                         self.global_network.variables)])

            # TODO write summaries
            # self.summary_writer = tf.summary.FileWriter('log' + "_%d" % self.task_index)
            if not self.optimizer:
                self.optimizer = tf.train.AdamOptimizer(self.alpha)

            else:
                optimizer_cls = get_function(self.optimizer)
                self.optimizer = optimizer_cls(self.alpha, *self.optimizer_args, **self.optimizer_kwargs)

            self.optimize_op = tf.group(self.optimizer.apply_gradients(grad_var_list),
                                     global_step_inc)

    def get_action(self, state, episode=1):
        return self.policy.sample(state)

    def update(self, batch):
        """
        Get global parameters, compute update, then send results to parameter server.
        :param batch:
        :return:
        """

        for episode in batch:
            episode['returns'] = discount(episode['rewards'], self.gamma)
            episode['advantages'] = self.generalised_advantage_estimation(episode)

        # Update linear value function for baseline prediction
        self.baseline_value_function.fit(batch)

        fetches = [self.loss, self.optimize_op, self.global_step]
        fetches.extend(self.local_network.internal_state_outputs)

        # Merge episode inputs into single arrays
        feed_dict = {
            self.episode_length: [episode['episode_length'] for episode in batch],
            self.state: [episode['states'] for episode in batch],
            self.actions: [episode['actions'] for episode in batch],
            self.advantage: [episode['advantages'] for episode in batch]
        }
        for n, internal_state in enumerate(self.local_network.internal_state_inputs):
            feed_dict[internal_state] = self.local_states[n]

        fetched = self.session.run(fetches, feed_dict)
        loss = fetched[0]
        self.local_states = fetched[3:]

        self.logger.debug('Distributed model loss = ' + str(loss))

    def get_global_step(self):
        """
        Returns global step to coordinator.
        :return:
        """
        return self.session.run(self.global_step)

    def sync_global_to_local(self):
        """
        Copy shared global weights to local network.

        """
        self.session.run(self.assign_global_to_local)

    def load_model(self, path):
        self.saver.restore(self.session, path)

    def save_model(self, path):
        self.saver.save(self.session, path)

    # TODO duplicate code -> refactor from pg model
    def generalised_advantage_estimation(self, episode):
        """
         Expects an episode, returns advantages according to config.
        """
        baseline = self.baseline_value_function.predict(episode)

        if self.generalized_advantage_estimation:
            if episode['terminated']:
                adjusted_baseline = np.append(baseline, [0])
            else:
                adjusted_baseline = np.append(baseline, baseline[-1])
            deltas = episode['rewards'] + self.gamma * adjusted_baseline[1:] - adjusted_baseline[:-1]
            advantage = discount(deltas, self.gamma * self.gae_lambda)
        else:
            advantage = episode['returns'] - baseline

        if self.normalize_advantage:
            return zero_mean_unit_variance(advantage)
        else:
            return advantage

    # TODO remove this duplicate when refactoring
    def zero_episode(self):
        """
        Creates a new episode dict.

        :return: 
        """
        zero_episode = {
            'episode_length': 0,
            'terminated': False,
            'states': np.zeros(shape=((self.batch_size,) + self.state_shape)),
            'actions': np.zeros(shape=(self.batch_size, self.action_count)),
            'action_means': np.zeros(shape=(self.batch_size, self.action_count)),
            'rewards': np.zeros(shape=(self.batch_size, 1))
        }

        if self.continuous:
            zero_episode['action_log_stds'] = np.zeros(shape=(self.batch_size, self.action_count))

        return zero_episode