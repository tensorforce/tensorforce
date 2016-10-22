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


import numpy as np

"""
Basic Reinforcement learning agent. An agent encapsulates execution logic
of a particular reinforcement learning algorithm and defines the external interface
to the environment. The agent hence acts an intermediate layer between environment
and backend execution (value function or policy updates).
"""


class RLAgent(object):
    pass

    def get_action(self, state):
        raise NotImplementedError

    def add_observation(self, state, action, reward, terminal):
        raise NotImplementedError

    def import_model(self, model_location):
        raise NotImplementedError

    def export_model(self, export_location):
        raise NotImplementedError
