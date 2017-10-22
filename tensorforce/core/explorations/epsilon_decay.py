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

from tensorforce.core.explorations import Exploration


class EpsilonDecay(Exploration):
    """
    Exponentially decaying epsilon parameter based on ratio of
    difference between current and final epsilon to total timesteps.
    """

    def __init__(self, epsilon=1.0, epsilon_final=0.1, epsilon_timesteps=10000, start_after=0, half_lives=10):
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_timesteps = epsilon_timesteps
        self.start_after = start_after
        self.half_life = self.epsilon_timesteps / half_lives

    def __call__(self, episode=0, timestep=0):
        if timestep < self.start_after:
            return self.epsilon

        offset = self.start_after

        if timestep > self.epsilon_timesteps:
            epsilon = self.epsilon_final
        else:
            epsilon = self.epsilon_final + \
                (self.epsilon - self.epsilon_final) * \
                2 ** (- (timestep - offset) / self.half_life)

        return epsilon
