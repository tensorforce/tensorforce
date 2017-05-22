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

from tensorforce.environments import Environment


class MinimalTest(Environment):

    def __init__(self, continuous):
        self.continuous = continuous

    def __str__(self):
        return 'MinimalTest'

    def close(self):
        pass

    def reset(self):
        self.state = (1.0, 0.0)
        return self.state

    def execute(self, action):
        if self.continuous:
            self.state = (self.state[0] - action, self.state[1] + action)
        else:
            if action == 0:
                self.state = (1.0, 0.0)
            elif action == 1:
                self.state = (0.0, 1.0)
            else:
                raise Exception()
        reward = self.state[1] * 2 - 1.0
        terminal = random() < 0.25
        return self.state, reward, terminal

    @property
    def states(self):
        return dict(shape=(2,), type='float')

    @property
    def actions(self):
        if self.continuous:
            return dict(continuous=True)
        else:
            return dict(continuous=False, num_actions=2)
