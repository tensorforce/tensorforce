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
Generic policy gradient agent.
"""
from tensorforce.config import create_config
from tensorforce.rl_agents.rl_agent import RLAgent


class PGAgent(RLAgent):

    default_config = {
        'batch_size': 10000,
        'episode_length': 100,
        'deterministic_mode': False,
        'max_kl': 0.01,
    }

    value_function_ref = None


    def __init__(self, config):

        self.config = create_config(config, default=self.default_config)
        self.updater = None

        if self.value_function_ref:
            self.updater = self.value_function_ref(self.config)

    def get_action(self, state, episode=1):
        """
        Executes one reinforcement learning step.

        :param state: Observed state tensor
        :param episode: Optional, current episode
        :return: Which action to take
        """

        return self.updater.get_action(state, episode)


    def add_observation(self, state, action, reward, terminal):
        """
        Adds an observation and performs a pg update if the necessary conditions
        are satisfied, i.e. if one batch of experience has been collected as defined
        by the batch size.
        :param state:
        :param action:
        :param reward:
        :param terminal:
        :return:
        """
        pass

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass