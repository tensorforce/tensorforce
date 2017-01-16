# Copyright 2016 reinforce.io. All Rights Reserved.
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
A policy gradient model provides generic methods used in pg algorithms, e.g.
GAE-computation or merging of episode data.
"""
from tensorforce.updater import Model
import numpy as np

from tensorforce.util.math_util import discount, zero_mean_unit_variance


class PGModel(Model):
    def __init__(self, config):
        super(PGModel, self).__init__(config)

        self.baseline_value_function = None

    def get_action(self, state, episode=1):
        pass

    def update(self, batch):
        pass

    def merge_episodes(self, batch):
        """
        Merge episodes of a batch into single input variables.

        :param batch:
        :return:
        """
        action_log_stds = np.concatenate([path['action_log_stds'] for path in batch])
        action_means = np.concatenate([path['action_means'] for path in batch])
        actions = np.concatenate([path['actions'] for path in batch])
        batch_advantage = np.concatenate([path["advantage"] for path in batch])
        batch_advantage = zero_mean_unit_variance(batch_advantage)
        states = np.concatenate([path['states'] for path in batch])

        return action_log_stds, action_means, actions, batch_advantage, states

    def compute_gae_advantage(self, batch, gamma, gae_lambda, use_gae=False):
        """
        Expects a batch containing at least one episode, sets advantages according to GAE.

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
