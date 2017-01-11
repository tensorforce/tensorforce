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
Implements and registers exploration strategies.
"""
import numpy as np

from tensorforce.util.experiment_util import global_seed


class Exploration():
    def __init__(self, deterministic_mode=False):

        if deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

    def get_noise(self, episode=0, states=0):
        pass


class OrnsteinUhlenbeckProcess(Exploration):
    def __init__(self, deterministic_mode, action_count=1, sigma=0.3, mu=0, theta=0.15):
        Exploration.__init__(self, deterministic_mode)
        self.action_count = action_count
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        if deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.state = np.ones(action_count) * self.mu

    def get_noise(self, episode=0, states=0):
        state = self.state
        dx = self.theta * (self.mu - state) + self.sigma * self.random.randn(len(state), 1)
        self.state = state + dx

        return self.state


class LinearDecay(Exploration):
    def __init__(self, deterministic_mode):
        Exploration.__init__(self, deterministic_mode)

    def get_noise(self, episode=0, states=0):
        return self.random.random_sample(1) / (episode + 1)


class ZeroExploration(Exploration):
    def __init__(self, deterministic_mode):
        Exploration.__init__(self, deterministic_mode)

    def get_noise(self, episode=None, states=None):
        return 0


class EpsilonDecay(Exploration):
    def __init__(self, deterministic_mode, epsilon=0.1, epsilon_final=0.1, epsilon_states=10000):
        Exploration.__init__(self, deterministic_mode)
        self.epsilon_final = epsilon_final
        self.epsilon = epsilon
        self.epsilon_states = epsilon_states

    def get_noise(self, episode=None, states=None):
        if states > self.epsilon_states:
            self.epsilon = self.epsilon_final
        else:
            self.epsilon += ((self.epsilon_final - self.epsilon) / self.epsilon_states) * states

        return self.epsilon


exploration_mode = {
    'None': ZeroExploration,
    'linear_decay': LinearDecay,
    'epsilon_decay': EpsilonDecay,
    'ornstein_uhlenbeck': OrnsteinUhlenbeckProcess
}
