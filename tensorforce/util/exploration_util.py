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


class Exploration(object):
    """
    Generic exploration object. Holds a reference to a agent to request
    shapes of action dimensions and deterministic mode.
    """
    def __init__(self, model=None):
        self.model = model
        if self.model.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

    def __call__(self, episode=0, states=0):
        pass


class OrnsteinUhlenbeckProcess(Exploration):
    def __init__(self, model, sigma=0.3, mu=0, theta=0.15):
        super(OrnsteinUhlenbeckProcess, self).__init__(model)
        self.action_count = self.model.action_count
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.state = np.ones(self.action_count) * self.mu

    def __call__(self, episode=0, states=0):
        state = self.state
        dx = self.theta * (self.mu - state) + self.sigma * self.random.randn(len(state), 1)
        self.state = state + dx

        return self.state


class LinearDecay(Exploration):
    """
    Linear decay based on episode number.
    """
    def __init__(self, model):
        super(LinearDecay, self).__init__(model)

    def __call__(self, episode=0, states=0):
        return self.random.random_sample(1) / (episode + 1)


class ConstantExploration(Exploration):
    """
    Constant exploration value, set to 0 if no configuration parameter is
    set.
    """
    def __init__(self, model, constant=0.):
        super(ConstantExploration, self).__init__(model)
        self.constant = constant

    def __call__(self, episode=None, states=None):
        return self.constant


class EpsilonDecay(Exploration):
    """
    Linearly decaying epsilon parameter based on number of states,
    an initial random epsilon and a final random epsilon.
    """
    def __init__(self, model, epsilon=0.1, epsilon_final=0.1, epsilon_states=10000):
        super(EpsilonDecay, self).__init__(model)
        self.epsilon_final = epsilon_final
        self.epsilon = epsilon
        self.epsilon_states = epsilon_states

    def __call__(self, episode=None, states=None):
        if states > self.epsilon_states:
            self.epsilon = self.epsilon_final
        else:
            self.epsilon += ((self.epsilon_final - self.epsilon) / self.epsilon_states) * states

        return self.epsilon


exploration_mode = {
    'constant': ConstantExploration,
    'linear_decay': LinearDecay,
    'epsilon_decay': EpsilonDecay,
    'ornstein_uhlenbeck': OrnsteinUhlenbeckProcess
}
