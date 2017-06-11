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

from tensorforce import util
from tensorforce.core.explorations import explorations


class Agent(object):

    name = None
    model = None
    default_config = dict()

    def __init__(self, config):
        assert self.__class__.name is not None and self.__class__.model is not None
        config.default(Agent.default_config)

        # only one state
        if 'type' in config.states:
            config.states = dict(state=config.states)
            self.unique_state = True
        else:
            config.states = config.states
            self.unique_state = False

        # only one action
        if 'continuous' in config.actions:
            config.actions = dict(action=config.actions)
            self.unique_action = True
        else:
            config.actions = config.actions
            self.unique_action = False

        self.states_config = config.states
        self.actions_config = config.actions

        self.model = self.__class__.model(config)

        # exploration
        self.exploration = dict()
        for name, action in config.actions:
            if 'exploration' not in action:
                self.exploration[name] = None
                continue
            exploration = action.exploration
            args = action.exploration_args
            kwargs = action.exploration_kwargs
            self.exploration[name] = util.function(exploration, explorations)(*args, **kwargs)

        self.episode = 0
        self.timestep = 0

    def __str__(self):
        return str(self.__class__.name)

    def reset(self):
        self.episode += 1
        self.internals = self.next_internals = self.model.reset()

    def act(self, state):
        self.timestep += 1
        self.internals = self.next_internals

        if self.unique_state:
            state = dict(state=state)

        action, self.next_internals = self.model.get_action(state=state, internals=self.internals)

        for name, exploration in self.exploration.items():
            if exploration is None:
                continue
            if self.actions_config[name].continuous:
                action[name] += exploration(episode=self.episode, timestep=self.timestep)
            else:
                if random() < exploration(episode=self.episode, timestep=self.timestep):
                    action[name] = randrange(self.actions_config[name].num_actions)
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
