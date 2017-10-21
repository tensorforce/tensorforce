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
            specification: Takes a list of (type, shape) pairs specifying the action structure of the environment.
        """
        self.specification = list()
        for action in specification:
            if len(action) == 2 and action[0] in ('bool', 'int', 'float', 'bounded-float'):
                if isinstance(action[1], int):
                    self.specification.append((action[0], (action[1],)))
                else:
                    self.specification.append((action[0], tuple(action[1])))
            else:
                raise TensorForceError('Invalid MinimalTest specification.')
        self.single_state_action = (len(specification) == 1)

    def __str__(self):
        return 'MinimalTest'

    def close(self):
        pass

    def reset(self):
        self.state = [(1.0, 0.0) for _ in self.specification]
        if self.single_state_action:
            return self.state[0]
        else:
            return {'state{}'.format(n): state for n, state in enumerate(self.state)}

    def execute(self, actions):
        if self.single_state_action:
            actions = (actions,)
        else:
            actions = tuple(actions[name] for name in sorted(actions))

        reward = 0.0
        for n, (action_type, shape) in enumerate(self.specification):
            if action_type == 'bool' or action_type == 'int':
                correct = np.sum(actions[n])
                overall = util.prod(shape)
                self.state[n] = ((overall - correct) / overall, correct / overall)
            elif action_type == 'float' or action_type == 'bounded-float':
                step = np.sum(actions[n]) / util.prod(shape)
                self.state[n] = max(self.state[n][0] - step, 0.0), min(self.state[n][1] + step, 1.0)
            reward += max(min(self.state[n][1], 1.0), 0.0)

        terminal = random() < 0.25
        if self.single_state_action:
            return self.state[0], terminal, reward
        else:
            reward = reward / len(self.specification)
            return {'state{}'.format(n): state for n, state in enumerate(self.state)}, terminal, reward

    @property
    def states(self):
        if self.single_state_action:
            return dict(shape=2, type='float')
        else:
            return {'state{}'.format(n): dict(shape=(2,), type='float') for n in range(len(self.specification))}

    @property
    def actions(self):
        if self.single_state_action:
            if self.specification[0][0] == 'int':
                return dict(type='int', num_actions=2)
            elif self.specification[0][0] == 'bounded-float':
                return dict(type='float', min_value=-0.5, max_value=1.5)
            else:
                return dict(type=self.specification[0][0])
        else:
            actions = dict()
            for n, (action_type, shape) in enumerate(self.specification):
                if action_type == 'int':
                    actions['action{}'.format(n)] = dict(type='int', shape=shape, num_actions=2)
                elif action_type == 'bounded-float':
                    actions['action{}'.format(n)] = dict(type='float', shape=shape, min_value=-0.5, max_value=1.5)
                else:
                    actions['action{}'.format(n)] = dict(type=action_type, shape=shape)
            return actions
