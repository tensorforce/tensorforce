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


class EpsilonAnneal(Exploration):
    """
    Annealing epsilon parameter based on ratio of current timestep to total timesteps.
    """

    def __init__(self, epsilon_start=1.0, epsilon_final=0.1, timesteps=10000, timestep_start=0):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.timesteps = timesteps
        self.timestep_start = timestep_start

    def __call__(self, episode=0, timestep=0):
        if timestep < self.timestep_start:
            return self.epsilon_start

        elif timestep > self.timestep_start + self.timesteps:
            return self.epsilon_final

        else:
            completed_ratio = (timestep - self.timestep_start) / self.timesteps
            return self.epsilon_start + completed_ratio * (self.epsilon_final - self.epsilon_start)
