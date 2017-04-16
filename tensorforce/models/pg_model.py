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
A policy gradient agent provides generic methods used in pg algorithms, e.g.
GAE-computation or merging of episode data.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorforce.models import Model
from tensorforce.models.baselines import LinearValueFunction, MLPValueFunction
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.models.policies import CategoricalOneHotPolicy
from tensorforce.models.policies import GaussianPolicy
from tensorforce.util.experiment_util import global_seed
from tensorforce.util.math_util import discount, zero_mean_unit_variance


class PGModel(Model):

    def __init__(self, config, scope, define_network=None):
        super(PGModel, self).__init__(config, scope)

        self.continuous = self.config.continuous
        self.batch_size = self.config.batch_size
        self.max_episode_length = min(self.config.max_episode_length, self.batch_size)
        self.action_count = self.config.actions

        # advantage estimation
        self.gamma = self.config.gamma
        self.generalized_advantage_estimation = self.config.gae
        self.gae_lambda = self.config.gae_lambda
        self.normalize_advantage = self.config.normalize_advantage

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.state_shape = tuple(self.config.state_shape)
        self.state = tf.placeholder(tf.float32, (None, None) + self.state_shape, name="state")
        self.actions = tf.placeholder(tf.float32, (None, None, self.action_count), name='actions')
        self.prev_action_means = tf.placeholder(tf.float32, (None, None, self.action_count), name='prev_actions')
        self.advantage = tf.placeholder(tf.float32, shape=(None, None, 1), name='advantage')

        if define_network is None:
            define_network = NeuralNetwork.layered_network(self.config.network_layers)
        if self.config.tf_scope is None:
            scope = ''
        else:
            scope = self.config.tf_scope + '-'
        self.network = NeuralNetwork(define_network, inputs=[self.state], episode_length=self.episode_length, scope=scope + 'value_function')
        self.internal_states = self.network.internal_state_inits

        # From an API perspective, continuous vs discrete might be easier than
        # requiring to set the concrete policy, at least currently
        if self.continuous:
            self.policy = GaussianPolicy(self.network, self.session, self.state, self.random, self.action_count, 'gaussian_policy')
            self.prev_action_log_stds = tf.placeholder(tf.float32, (None, self.batch_size, self.action_count))
            self.prev_dist = dict(
                policy_output=self.prev_action_means,
                policy_log_std=self.prev_action_log_stds)

        else:
            self.policy = CategoricalOneHotPolicy(self.network, self.session, self.state, self.random, self.action_count, 'categorical_policy')
            self.prev_dist = dict(policy_output=self.prev_action_means)

        # Probability distribution used in the current policy
        self.dist = self.policy.get_distribution()

        self.baseline_value_function = LinearValueFunction()
       # self.saver = tf.train.Saver()

    def get_action(self, state, episode=1):
        """
        Actions are directly sampled from the policy.

        :param state:
        :param episode:
        :return:
        """

        return self.policy.sample(state)

    def update(self, batch):
        """
        Update needs to be implemented by specific PG algorithm.

        :param batch: Batch of experiences
        :return:
        """
        raise NotImplementedError

    def zero_episode(self):
        """
        Creates a new episode dict.
        
        :return: 
        """
        zero_episode = {
            'episode_length': 0,
            'terminated': False,
            'states': np.zeros(shape=((self.max_episode_length,) + self.state_shape)),
            'actions': np.zeros(shape=(self.max_episode_length, self.action_count)),
            'action_means': np.zeros(shape=(self.max_episode_length, self.action_count)),
            'rewards': np.zeros(shape=(self.max_episode_length, 1))
        }

        if self.continuous:
            zero_episode['action_log_stds'] = np.zeros(shape=(self.max_episode_length, self.action_count))

        return zero_episode

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
