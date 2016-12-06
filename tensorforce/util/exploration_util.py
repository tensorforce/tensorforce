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
Implements and registers exploration strategies for continuous control problems.
"""
import numpy as np


# TODO implement ornstein-uhlenbeck process

def linear_decay(random, episode):
    return random.random_sample(1) / (episode + 1)


def zero(random=None, episode=None):
    return 0


def epsilon_decay(epsilon_final, total_states, epsilon_states, epsilon):
    if not epsilon_final or total_states == 0:
        epsilon = epsilon
    elif total_states > epsilon_states:
        epsilon = epsilon_final
    else:
        epsilon = epsilon + ((epsilon_final - epsilon) / epsilon_states) * total_states

    return epsilon


exploration_mode = {
    'None': zero,
    'linear_decay': linear_decay,
    'epsilon_decay': epsilon_decay
}
