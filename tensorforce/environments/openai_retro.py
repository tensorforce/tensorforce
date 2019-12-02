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

from tensorforce.environments import OpenAIGym


class OpenAIRetro(OpenAIGym):
    """
    [OpenAI Retro](https://github.com/openai/retro) environment adapter (specification key:
    `retro`, `openai_retro`).

    May require:
    ```bash
    pip3 install gym-retro
    ```

    Args:
        level (string): Game id
            (<span style="color:#C00000"><b>required</b></span>).
        visualize (bool): Whether to visualize interaction
            (<span style="color:#00C000"><b>default</b></span>: false).
        monitor_directory (string): Monitor output directory
            (<span style="color:#00C000"><b>default</b></span>: none).
        kwargs: Additional Retro environment arguments.
    """

    @classmethod
    def levels(cls):
        import retro

        return list(retro.data.list_games())

    @classmethod
    def create_level(cls, level, max_episode_steps, reward_threshold, tags, **kwargs):
        import retro

        assert max_episode_steps is False and reward_threshold is None and tags is None

        return retro.make(game=level, **kwargs), max_episode_steps

    def __init__(self, level, visualize=False, visualize_directory=None, **kwargs):
        import retro

        super().__init__(
            level=level, visualize=visualize, max_episode_steps=False, terminal_reward=0.0,
            reward_threshold=None, tags=None, visualize_directory=visualize_directory, **kwargs
        )
