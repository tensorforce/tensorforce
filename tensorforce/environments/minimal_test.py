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

    def __init__(self, definition):
        if isinstance(definition, bool):
            self.definition = [(definition, ())]
            self.single_state_action = True
        else:
            self.definition = list()
            for action in definition:
                if isinstance(action, bool):
                    self.definition.append((action, ()))

                elif len(action) == 2:
                    if isinstance(action[1], int):
                        self.definition.append((action[0], (action[1],)))
                    else:
                        self.definition.append((action[0], tuple(action[1])))
                else:
                    raise TensorForceError('Invalid MinimalTest definition.')
            self.single_state_action = False

    def __str__(self):
        return 'MinimalTest'

    def close(self):
        pass

    def reset(self):
        self.state = [(1.0, 0.0) for _ in self.definition]
        if self.single_state_action:
            return self.state[0]
        else:
            return {'state{}'.format(n): state for n, state in enumerate(self.state)}

    def execute(self, action):
        if self.single_state_action:
            action = (action,)
        else:
            action = tuple(action[name] for name in sorted(action))

        reward = 0.0
        for n, (continuous, shape) in enumerate(self.definition):
            if continuous:
                step = np.sum(action[n]) / util.prod(shape)
                self.state[n] = (max(self.state[n][0] - step, 0.0), min(self.state[n][1] + step, 1.0))
            else:
                correct = np.sum(action[n])
                overall = util.prod(shape)
                self.state[n] = ((overall - correct) / overall, correct / overall)
            reward += self.state[n][1] * 2 - 1.0

        terminal = random() < 0.25
        if self.single_state_action:
            return self.state[0], reward, terminal
        else:
            reward = reward / len(self.definition)
            return {'state{}'.format(n): state for n, state in enumerate(self.state)}, reward, terminal

    @property
    def states(self):
        if self.single_state_action:
            return dict(shape=2, type='float')
        else:
            return {'state{}'.format(n): dict(shape=(2,), type='float') for n in range(len(self.definition))}

    @property
    def actions(self):
        if self.single_state_action:
            if self.definition[0][0]:
                return dict(continuous=True)
            else:
                return dict(continuous=False, num_actions=2)
        else:
            actions = dict()
            for n, (continuous, shape) in enumerate(self.definition):
                if continuous:
                    actions['action{}'.format(n)] = dict(continuous=True, shape=shape)
                else:
                    actions['action{}'.format(n)] = dict(continuous=False, shape=shape, num_actions=2)
            return actions
