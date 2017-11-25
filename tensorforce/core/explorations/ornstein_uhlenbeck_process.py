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


class OrnsteinUhlenbeckProcess(Exploration):
    """
    Explore via an Ornstein-Uhlenbeck process.
    """

    def __init__(self, sigma=0.3, mu=0, theta=0.15):
        """
        Initializes an Ornstein-Uhlenbeck process which is a mean reverting stochastic process
        introducing time-correlated noise.
        """
        self.sigma = sigma
        self.mu = self.state = mu
        self.theta = theta

    def __call__(self, episode=0, timestep=0, num_actions=1):
        normal_sample = tf.random_normal(
            shape=num_actions,
            mean=0.0,
            stddev=1.0
        )
        state = tf.get_variable(
            name='ornstein_uhlenbeck',
            shape=num_actions,
            initializer=tf.constant_initializer(self.mu)
        )
        self.state = tf.assign_add(
            ref=state,
            value=self.theta * (self.mu - self.state) + self.sigma * normal_sample
        )

        return self.state
