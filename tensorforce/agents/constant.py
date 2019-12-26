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

from collections import OrderedDict

from tensorforce.agents import Agent
from tensorforce.core.models import ConstantModel


class ConstantAgent(Agent):
    """
    Agent returning constant action values (specification key: `constant`).

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

        action_values (dict[value]): Constant value per action
            (<span style="color:#00C000"><b>default</b></span>: false for binary boolean actions,
            0 for discrete integer actions, 0.0 for continuous actions).
        seed (int): Random seed to set for Python, NumPy (both set globally!) and TensorFlow,
            environment seed has to be set separately for a fully deterministic execution
            (<span style="color:#00C000"><b>default</b></span>: none).
        name (string): Agent name, used e.g. for TensorFlow scopes
            (<span style="color:#00C000"><b>default</b></span>: "agent").
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: TensorFlow default).
        summarizer (specification): TensorBoard summarizer configuration with the following
            attributes (<span style="color:#00C000"><b>default</b></span>: no summarizer):
            <ul>
            <li><b>directory</b> (<i>path</i>) &ndash; summarizer directory
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>frequency</b> (<i>int > 0) &ndash; how frequently in timesteps to record
            summaries (<span style="color:#00C000"><b>default</b></span>: always).</li>
            <li><b>flush</b> (<i>int > 0</i>) &ndash; how frequently in seconds to flush the
            summary writer (<span style="color:#00C000"><b>default</b></span>: 10).</li>
            <li><b>max-summaries</b> (<i>int > 0</i>) &ndash; maximum number of summaries to keep
            (<span style="color:#00C000"><b>default</b></span>: 5).</li>
            <li><b>labels</b> (<i>"all" | iter[string]</i>) &ndash; all or list of summaries to
            record, from the following labels
            (<span style="color:#00C000"><b>default</b></span>: only "graph"):</li>
            <li>"graph": graph summary</li>
            <li>"parameters": parameter scalars</li>
            </ul>
        recorder (specification): Experience traces recorder configuration with the following
            attributes (<span style="color:#00C000"><b>default</b></span>: no recorder):
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
        # Agent
        action_values=None,
        # TensorFlow etc
        name='agent', device=None, seed=None, summarizer=None, recorder=None, config=None
    ):
        self.spec = OrderedDict(
            agent='constant',
            states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
            action_values=action_values,
            name=name, device=device, seed=seed, summarizer=summarizer, recorder=recorder,
            config=config
        )

        super().__init__(
            states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
            parallel_interactions=1, buffer_observe=True, seed=seed, recorder=recorder
        )

        self.model = ConstantModel(
            # Model
            name=name, device=device, parallel_interactions=self.parallel_interactions,
            buffer_observe=self.buffer_observe, seed=seed, summarizer=summarizer, config=config,
            states=self.states_spec, actions=self.actions_spec,
            # ConstantModel
            action_values=action_values
        )
