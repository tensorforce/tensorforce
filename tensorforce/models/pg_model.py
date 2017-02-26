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
import tensorflow as tf
from tensorforce.models import Model
import numpy as np

from tensorforce.models.baselines.linear_value_function import LinearValueFunction
from tensorforce.models.baselines.mlp_value_function import MLPValueFunction
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.models.policies import CategoricalOneHotPolicy
from tensorforce.models.policies import GaussianPolicy
from tensorforce.util.experiment_util import global_seed
from tensorforce.util.math_util import discount, zero_mean_unit_variance


class PGModel(Model):
    def __init__(self, config, scope):
        super(PGModel, self).__init__(config, scope)
        self.batch_size = self.config.batch_size
        self.action_count = self.config.actions
        self.use_gae = self.config.use_gae
        self.gae_lambda = self.config.gae_lambda

        self.gamma = self.config.gamma
        self.continuous = self.config.continuous
        self.normalize_advantage = self.config.normalise_advantage

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")
        self.episode = 0
        self.input_feed = None

        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
        self.policy = None

        scope = '' if self.config.tf_scope is None else self.config.tf_scope + '-'
        self.hidden_layers = NeuralNetwork(self.config.network_layers, self.state,
                                           scope=scope + 'value_function')

        self.saver = tf.train.Saver()
        self.actions = tf.placeholder(tf.float32, [None, self.action_count], name='actions')
        self.prev_action_means = tf.placeholder(tf.float32, [None, self.action_count], name='prev_actions')

        # From an API perspective, continuous vs discrete might be easier than
        # requiring to set the concrete policy, at least currently
        if self.continuous:
            self.policy = GaussianPolicy(self.hidden_layers, self.session, self.state, self.random,
                                         self.action_count, 'gaussian_policy')
            self.prev_action_log_stds = tf.placeholder(tf.float32, [None, self.action_count])

            self.prev_dist = dict(policy_output=self.prev_action_means,
                                  policy_log_std=self.prev_action_log_stds)

        else:
            self.policy = CategoricalOneHotPolicy(self.hidden_layers, self.session, self.state, self.random,
                                                  self.action_count, 'categorical_policy')
            self.prev_dist = dict(policy_output=self.prev_action_means)

        # Probability distribution used in the current policy
        self.dist = self.policy.get_distribution()

        # TODO configurable value functions
        self.baseline_value_function = MLPValueFunction(self.session, 100, 64)

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

    def merge_episodes(self, batch):
        """
        Merge episodes of a batch into single input variables.

        :param batch:
        :return:
        """
        if self.continuous:
            action_log_stds = np.concatenate([path['action_log_stds'] for path in batch])
            action_log_stds = np.expand_dims(action_log_stds, axis=1)
        else:
            action_log_stds = None

        action_means = np.concatenate([path['action_means'] for path in batch])
        actions = np.concatenate([path['actions'] for path in batch])
        batch_advantage = np.concatenate([path["advantage"] for path in batch])

        if self.normalize_advantage:
            batch_advantage = zero_mean_unit_variance(batch_advantage)

        batch_advantage = np.expand_dims(batch_advantage, axis=1)
        states = np.concatenate([path['states'] for path in batch])

        return action_log_stds, action_means, actions, batch_advantage, states

    def compute_gae_advantage(self, batch, gamma, gae_lambda, use_gae=False):
        """
        Expects a batch containing at least one episode, sets advantages according to use_gae.

        :param batch: Sequence of observations for at least one episode.
        """

        for episode in batch:
            baseline = self.baseline_value_function.predict(episode)

            if episode['terminated']:
                adjusted_baseline = np.append(baseline, [0])
            else:
                adjusted_baseline = np.append(baseline, baseline[-1])

            episode['returns'] = discount(episode['rewards'], gamma)

            if use_gae:
                deltas = episode['rewards'] + gamma * adjusted_baseline[1:] - adjusted_baseline[:-1]
                episode['advantage'] = discount(deltas, gamma * gae_lambda)
            else:
                episode['advantage'] = episode['returns'] - baseline
