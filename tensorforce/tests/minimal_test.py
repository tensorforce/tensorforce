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

from random import random

import numpy as np

from tensorforce import util, TensorForceError
from tensorforce.environments import Environment


class MinimalTest(Environment):

    def __init__(self, specification):
        """
        Initializes a minimal test environment, which is used for the unit tests.
        Given a specification of actions types and shapes, the environment states consist
        of the same number of pairs (x, y). The (mean of) an action a gives the next state via (1-a, a),
        and the 'correct' state is always (0, 1).

        Args:
            specification: Takes a dict type (keys)-> shape (values specifying the action
                structure of the environment. Use shape () for single scalar actions.
        """
        self.specification = dict()
        for action_type, shape in specification.items():
            if action_type in ('bool', 'int', 'float', 'bounded'):
                if isinstance(shape, int):
                    self.specification[action_type] = (shape,)
                else:
                    self.specification[action_type] = tuple(shape)
            else:
                raise TensorForceError('Invalid MinimalTest specification.')
        self.single_state_action = (len(specification) == 1)

    def __str__(self):
        return 'MinimalTest'

    def close(self):
        pass

    def reset(self):
        self.state = {action_type: (1.0, 0.0) for action_type in self.specification}
        if self.single_state_action:
            return next(iter(self.state.values()))
        else:
            return dict(self.state)

    def execute(self, action):
        if self.single_state_action:
            action = {next(iter(self.specification)): action}

        reward = 0.0
        for action_type, shape in self.specification.items():
            if action_type == 'bool' or action_type == 'int':
                correct = np.sum(action[action_type])
                overall = util.prod(shape)
                self.state[action_type] = ((overall - correct) / overall, correct / overall)
            elif action_type == 'float' or action_type == 'bounded':
                step = np.sum(action[action_type]) / util.prod(shape)
                self.state[action_type] = max(self.state[action_type][0] - step, 0.0), min(self.state[action_type][1] + step, 1.0)
            reward += max(min(self.state[action_type][1], 1.0), 0.0)

        terminal = random() < 0.25
        if self.single_state_action:
            return next(iter(self.state.values())), terminal, reward
        else:
            reward = reward / len(self.specification)
            return dict(self.state), terminal, reward

    @property
    def states(self):
        if self.single_state_action:
            return dict(shape=2, type='float')
        else:
            return {action_type: dict(shape=(2,), type='float') for action_type in self.specification}

    @property
    def actions(self):
        if self.single_state_action:
            action_type = next(iter(self.specification))
            if action_type == 'int':
                return dict(type='int', num_actions=2)
            elif action_type == 'bounded':
                return dict(type='float', min_value=-0.5, max_value=1.5)
            else:
                return dict(type=action_type)
        else:
            actions = dict()
            for action_type, shape in self.specification.items():
                if action_type == 'int':
                    actions[action_type] = dict(type='int', shape=shape, num_actions=2)
                elif action_type == 'bounded':
                    actions[action_type] = dict(type='float', shape=shape, min_value=-0.5, max_value=1.5)
                else:
                    actions[action_type] = dict(type=action_type, shape=shape)
            return actions
