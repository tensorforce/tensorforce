# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from tensorforce.environments import Environment


class OpenSim(Environment):
    """
    [OpenSim](http://osim-rl.stanford.edu/) environment adapter (specification key: `osim`,
    `open_sim`).

    Args:
        level ('Arm2D' | 'L2Run' | 'Prosthetics'): Environment id
            (<span style="color:#C00000"><b>required</b></span>).
        visualize (bool): Whether to visualize interaction
            (<span style="color:#00C000"><b>default</b></span>: false).
        integrator_accuracy (float): Integrator accuracy
            (<span style="color:#00C000"><b>default</b></span>: 5e-5).
    """

    @classmethod
    def levels(cls):
        return ['Arm2D', 'L2Run', 'Prosthetics']

    def __init__(self, level, visualize=False, integrator_accuracy=5e-5):
        super().__init__()

        from osim.env import L2RunEnv, Arm2DEnv, ProstheticsEnv

        environments = dict(Arm2D=Arm2DEnv, L2Run=L2RunEnv, Prosthetics=ProstheticsEnv)

        self.environment = environments[level](
            visualize=visualize, integrator_accuracy=integrator_accuracy
        )

    def __str__(self):
        return super().__str__() + '({})'.format(self.environment)

    def states(self):
        return dict(type='float', shape=self.environment.get_observation_space_size())

    def actions(self):
        return dict(type='float', shape=self.environment.get_action_space_size())

    def close(self):
        self.environment.close()
        self.environment = None

    def reset(self):
        return self.environment.reset()

    def execute(self, actions):
        states, reward, terminal, _ = self.env.step(action=actions)
        return states, terminal, reward
