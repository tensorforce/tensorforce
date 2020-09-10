# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from collections import OrderedDict

from tensorforce.agents import Agent
from tensorforce.core.models import RandomModel


class RandomAgent(Agent):
    """
    Agent returning random action values (specification key: `random`).

    Args:
        states (specification): States specification
            (<span style="color:#C00000"><b>required</b></span>, better implicitly specified via
            `environment` argument for `Agent.create(...)`), arbitrarily nested dictionary of state
            descriptions (usually taken from `Environment.states()`) with the following attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; state data type
            (<span style="color:#00C000"><b>default</b></span>: "float").</li>
            <li><b>shape</b> (<i>int | iter[int]</i>) &ndash; state shape
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>num_values</b> (<i>int > 0</i>) &ndash; number of discrete state values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum state value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        actions (specification): Actions specification
            (<span style="color:#C00000"><b>required</b></span>, better implicitly specified via
            `environment` argument for `Agent.create(...)`), arbitrarily nested dictionary of
            action descriptions (usually taken from `Environment.actions()`) with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; action data type
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>shape</b> (<i>int > 0 | iter[int > 0]</i>) &ndash; action shape
            (<span style="color:#00C000"><b>default</b></span>: scalar).</li>
            <li><b>num_values</b> (<i>int > 0</i>) &ndash; number of discrete action values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum action value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        max_episode_timesteps (int > 0): Upper bound for numer of timesteps per episode
            (<span style="color:#00C000"><b>default</b></span>: not given, better implicitly
            specified via `environment` argument for `Agent.create(...)`).

        config (specification): Additional configuration options:
            <ul>
            <li><b>name</b> (<i>string</i>) &ndash; Agent name, used e.g. for TensorFlow scopes
            (<span style="color:#00C000"><b>default</b></span>: "agent").
            <li><b>device</b> (<i>string</i>) &ndash; Device name
            (<span style="color:#00C000"><b>default</b></span>: TensorFlow default).
            <li><b>seed</b> (<i>int</i>) &ndash; Random seed to set for Python, NumPy (both set
            globally!) and TensorFlow, environment seed may have to be set separately for fully
            deterministic execution
            (<span style="color:#00C000"><b>default</b></span>: none).</li>
            <li><b>buffer_observe</b> (<i>false | "episode" | int > 0</i>) &ndash; Number of
            timesteps within an episode to buffer before calling the internal observe function, to
            reduce calls to TensorFlow for improved performance
            (<span style="color:#00C000"><b>default</b></span>: configuration-specific maximum
            number which can be buffered without affecting performance).</li>
            <li><b>always_apply_exploration</b> (<i>bool</i>) &ndash; Whether to always apply
            exploration, also for independent `act()` calls (final value in case of schedule)
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            <li><b>always_apply_variable_noise</b> (<i>bool</i>) &ndash; Whether to always apply
            variable noise, also for independent `act()` calls (final value in case of schedule)
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            <li><b>enable_int_action_masking</b> (<i>bool</i>) &ndash; Whether int action options
            can be masked via an optional "[ACTION-NAME]_mask" state input
            (<span style="color:#00C000"><b>default</b></span>: true).</li>
            <li><b>create_tf_assertions</b> (<i>bool</i>) &ndash; Whether to create internal
            TensorFlow assertion operations
            (<span style="color:#00C000"><b>default</b></span>: true).</li>
            </ul>
        recorder (path | specification): Traces recordings directory, or recorder configuration with
            the following attributes (see
            [record-and-pretrain script](https://github.com/tensorforce/tensorforce/blob/master/examples/record_and_pretrain.py)
            for example application)
            (<span style="color:#00C000"><b>default</b></span>: no recorder):
            <ul>
            <li><b>directory</b> (<i>path</i>) &ndash; recorder directory
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>frequency</b> (<i>int > 0</i>) &ndash; how frequently in episodes to record
            traces (<span style="color:#00C000"><b>default</b></span>: every episode).</li>
            <li><b>start</b> (<i>int >= 0</i>) &ndash; how many episodes to skip before starting to
            record traces (<span style="color:#00C000"><b>default</b></span>: 0).</li>
            <li><b>max-traces</b> (<i>int > 0</i>) &ndash; maximum number of traces to keep
            (<span style="color:#00C000"><b>default</b></span>: all).</li>
    """

    def __init__(
        # Environment
        self, states, actions, max_episode_timesteps=None,
        # Config, recorder
        config=None, recorder=None
    ):
        if not hasattr(self, 'spec'):
            self.spec = OrderedDict(
                agent='random',
                states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
                config=config, recorder=recorder
            )

        super().__init__(
            states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
            parallel_interactions=1, config=config, recorder=recorder
        )

        self.model = RandomModel(
            states=self.states_spec, actions=self.actions_spec,
            parallel_interactions=self.parallel_interactions,
            config=self.config, summarizer=None
        )
