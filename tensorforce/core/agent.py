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
Basic Reinforcement learning agent. An agent encapsulates execution logic
of a particular reinforcement learning algorithm and defines the external interface
to the environment.

The agent hence acts an intermediate layer between environment
and backend execution (value function or policy updates).
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from random import random, randrange

from tensorforce.util import module
from tensorforce.core.explorations import explorations


class Agent(object):

    name = None
    model = None
    default_config = dict()

    def __init__(self, config, network_builder):
        assert self.__class__.name is not None and self.__class__.model is not None
        config.default(Agent.default_config)

        # if only one state/action
        if 'type' in config.states:
            config.states = dict(state=config.states)
            self.unique_state = True
        else:
            self.unique_state = False
        if 'continuous' in config.actions:
            config.actions = dict(action=config.actions)
            self.unique_action = True
        else:
            self.unique_action = False

        self.state_config = config.states
        self.action_config = config.actions

        self.model = self.__class__.model(config, network_builder)

        # exploration
        self.exploration = dict()
        for name, action in self.action_config.items():
            exploration = action.get('exploration', None)
            args = action.get('exploration_args', ())
            kwargs = action.get('exploration_kwargs', {})
            if exploration is None:
                self.exploration[name] = None
            elif exploration in explorations:
                self.exploration[name] = explorations[exploration](*args, **kwargs)
            else:
                self.exploration[name] = module(exploration)(*args, **kwargs)

        self.episodes = 0
        self.timesteps = 0

    def __str__(self):
        return str(self.__class__.name)

    def reset(self):
        self.episodes += 1
        self.model.reset()

    def act(self, state):
        self.timesteps += 1
        if self.unique_state:
            state = dict(state=state)
        action, self.internals = self.model.get_action(state=state)
        for name, exploration in self.exploration.items():
            if exploration is None:
                continue
            if name in self.discrete_actions:
                if random() < exploration(episodes=self.episodes, timesteps=self.timesteps):
                    action[name] = randrange(high=self.action_config[name]['num_actions'])
            else:
                action[name] += exploration(episodes=self.episodes, timesteps=self.timesteps)
        if self.unique_action:
            return action['action']
        else:
            return action

    def observe(self, state, action, reward, terminal):
        raise NotImplementedError

    def load_model(self, path):
        self.model.load_model(path)

    def save_model(self, path):
        self.model.save_model(path)
