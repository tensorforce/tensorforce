# Copyright 2021 Tensorforce Team. All Rights Reserved.
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

from tensorforce import Environment, Runner


class EnvironmentWithMasking(Environment):
    """
    States: {0, 1, ..., 9, 10}
    Actions: {-1, 0, 1}
    Action masking: action = -1 invalid for state = 0, action = 1 invalid for state = 10
    Reward:
        - Positive: [state < 5, action = 1] or [state > 5, action = -1]
        - Negative: [state < 5, action = -1] or [state > 5, action = 1]
    """

    def __init__(self):
        super().__init__()

    def states(self):
        # States specification does not need to include action mask item
        return dict(type=int, shape=(), num_values=11)

    def actions(self):
        # Only discrete actions can be masked
        return dict(type=int, shape=(), num_values=3)

    def reset(self):
        # Initial state and associated action mask
        self.state = np.random.randint(3, 7)
        action_mask = np.asarray([self.state > 0, True, self.state < 10])

        # Add action mask to states dictionary (mask item is "[NAME]_mask", here "action_mask")
        states = dict(state=self.state, action_mask=action_mask)

        return states

    def execute(self, actions):
        # Compute terminal and reward
        terminal = False
        if actions == 1:
            reward = -np.abs(self.state / 5.0 - 1.0)
        else:
            reward = (1 - actions) * (self.state / 5.0 - 1.0)

        # Compute next state and associated action mask
        self.state += actions - 1
        action_mask = np.asarray([self.state > 0, True, self.state < 10])

        # Add action mask to states dictionary (mask item is "[NAME]_mask", here "action_mask")
        states = dict(state=self.state, action_mask=action_mask)

        return states, terminal, reward


if __name__ == '__main__':
    agent = 'benchmarks/configs/ppo.json'
    runner = Runner(agent=agent, environment=EnvironmentWithMasking, max_episode_timesteps=20)
    runner.run(num_episodes=100)
    runner.close()
