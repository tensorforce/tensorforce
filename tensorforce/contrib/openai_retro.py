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

import retro

from tensorforce import TensorForceError
from tensorforce.contrib.openai_gym import OpenAIGym


class OpenAIRetro(OpenAIGym):
    """
    OpenAI Retro environment (https://github.com/openai/retro).
    Requires installation via `pip install gym-retro`.
    """

    def __init__(self, game, state=None):
        self.game = game
        if state is None:
            self.gym = retro.make(game)
        else:
            self.gym = retro.make(game, state=state)
        self.visualize = False

        self.states_spec = OpenAIGym.specs_from_gym_space(
            space=self.gym.observation_space, ignore_value_bounds=True
        )
        self.actions_spec = OpenAIGym.specs_from_gym_space(
            space=self.gym.action_space, ignore_value_bounds=False
        )

    def __str__(self):
        return 'OpenAIRetro({})'.format(self.game)
