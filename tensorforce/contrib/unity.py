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

"""
Unity ML Agents Integration: https://github.com/Unity-Technologies/ml-agents.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
from mlagents.envs import UnityEnvironment
from tensorforce import TensorForceError
from tensorforce.environments import Environment

class UnityEnv(Environment):
    """
    Bindings for Unity ML Agents https://github.com/Unity-Technologies/ml-agents
    For installation instruction see https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
    """

    def __init__(self, env, worker_id=0):
        """
        Initialize Unity Env.

        Args:
            env: Path to pre-built unity environment binary file. Give path to the folder containing the
            binary file followed by environment name.
            worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.

        """

        self.env = UnityEnvironment(file_name=env, worker_id=worker_id)
        UnityEnv.check_compatibility(self.env)
        default_brain = UnityEnv.get_brain_by_index(self.env, 0)
        self._states = UnityEnv.state_from_brain(default_brain)
        self._actions = UnityEnv.action_from_brain(default_brain)

    def __str__(self):
        return 'Unity Environment({})'.format(self.env)

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    def close(self):
        self.env.close()

    def reset(self):
        env_info = self.env.reset()
        default_brain_index, default_agent_index = 0, 0
        env_info_default = UnityEnv.get_env_info_by_brain(self.env, env_info, default_brain_index)
        state = env_info_default.vector_observations[default_agent_index].flatten()
        return state

    def execute(self, action):
        env_info = self.env.step(action)
        default_brain_index, default_agent_index = 0, 0 #TODO: add multi brain, multi agent support
        env_info_default = UnityEnv.get_env_info_by_brain(self.env, env_info, default_brain_index)
        state = env_info_default.vector_observations[default_agent_index].flatten()
        reward = env_info_default.rewards[default_agent_index]
        done = env_info_default.local_done[default_agent_index]
        return state, done, reward

    @staticmethod
    def get_env_info_by_brain(env, env_info, brain_index):
        brain_name = env.brain_names[brain_index]
        return env_info[brain_name]

    @staticmethod
    def get_brain_by_index(env, brain_index):
        brain_name = env.brain_names[brain_index]
        brain = env.brains[brain_name]
        return brain

    @staticmethod
    def check_compatibility(env):
        if env.number_brains > 1:
            logging.warning("Tensorforce currently doesn't support multiple brains, only default "
                            "brain {} will be trained".format(env.brain_names[0]))
        env_info = UnityEnv.get_env_info_by_brain(env, env.reset(), 0)
        n_agents = env_info.vector_observations.shape[0]
        if n_agents > 1:
            raise TensorForceError("The provided unity environment has {} agents. Tensorforce currently "
                                   "doesn't support getting observations from multiple agents".format(n_agents))

    @staticmethod
    def state_from_brain(brain):
        return dict(shape=(brain.vector_observation_space_size,), type='float')

    @staticmethod
    def action_from_brain(brain):
        if brain.vector_action_space_type=="discrete":
            return dict(type='int', num_actions=brain.vector_action_space_size[0])
        else:
            return dict(shape=(brain.vector_action_space_size[0],), type='float')
