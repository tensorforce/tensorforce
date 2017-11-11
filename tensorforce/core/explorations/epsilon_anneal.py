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

    def __init__(self, initial_epsilon=1.0, final_epsilon=0.1, timesteps=10000, start_timestep=0):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.timesteps = timesteps
        self.start_timestep = start_timestep

    def __call__(self, episode=0, timestep=0):
        if timestep < self.start_timestep:
            return self.initial_epsilon

        elif timestep > self.start_timestep + self.timesteps:
            return self.final_epsilon

        else:
            completed_ratio = (timestep - self.start_timestep) / self.timesteps
            return self.initial_epsilon + completed_ratio * (self.final_epsilon - self.initial_epsilon)
