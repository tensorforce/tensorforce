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


class MultiactorEnvironment(Environment):
    """
    Example multi-actor environment, illustrating best-practice implementation pattern.

    State space: position in [0, 10].
    Action space: movement in {-1, 0, 1}.
    Random start in [3, 7].
    Actor 1 perspective as is, actor 2 perspective mirrored.
    Positive reward for being closer to 10.
    """

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='int', num_values=11)

    def actions(self):
        return dict(type='int', num_values=3)

    def num_actors(self):
        return 2  # Indicates that environment has multiple actors

    def reset(self):
        # Always for multi-actor environments: initialize parallel indices
        self._parallel_indices = np.arange(self.num_actors())

        # Single shared environment logic, plus per-actor perspective
        self._states = 3 + np.random.randint(5)
        self.second_actor = True
        states = np.stack([self._states, 10 - self._states], axis=0)

        # Always for multi-actor environments: return per-actor values
        return self._parallel_indices.copy(), states

    def execute(self, actions):
        # Single shared environment logic, plus per-actor perspective
        if self.second_actor:
            self.second_actor = self.second_actor and not (np.random.random_sample() < 0.1)
            terminal = np.stack([False, not self.second_actor], axis=0)
            delta = (actions[0] - 1) - (actions[1] - 1)
            self._states = np.clip(self._states + delta, a_min=0, a_max=10)
            states = np.stack([self._states, 10 - self._states], axis=0)
        else:
            terminal = np.stack([False], axis=0)
            delta = (actions[0] - 1)
            self._states = np.clip(self._states + delta, a_min=0, a_max=10)
            states = np.stack([self._states], axis=0)
        reward = (states - 5.0) / 5.0

        # Always for multi-actor environments: update parallel indices, and return per-actor values
        self._parallel_indices = self._parallel_indices[~terminal]
        return self._parallel_indices.copy(), states, terminal, reward


def main():
    # Multi-actor runner, automatically if environment.num_actors() > 1
    runner = Runner(
        agent='benchmarks/configs/ppo.json',
        environment=MultiactorEnvironment,
        max_episode_timesteps=10
    )
    runner.run(num_episodes=1000)


if __name__ == '__main__':
    main()
