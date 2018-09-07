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

import tensorflow as tf

from tensorforce.core.explorations import Exploration


class GaussianNoise(Exploration):
    """
    Explores via gaussian noise.
    """

    def __init__(
        self,
        sigma=0.3,
        mu=0.0,
        scope='gaussian_noise',
        summary_labels=()
    ):
        """
        Initializes distribution values for gaussian noise
        """
        self.sigma = sigma
        self.mu = float(mu)  # need to add cast to float to avoid tf type-mismatch error in case mu=0.0

        super(GaussianNoise, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_explore(self, episode, timestep, shape):
        return tf.random_normal(shape=shape, mean=self.mu, stddev=self.sigma)
