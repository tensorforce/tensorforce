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
        states (specification): States specification (**required**), arbitrarily nested
            dictionary of state descriptions (usually taken from `Environment.states()`) with
            the following attributes:
            <ul>
            <li><b>type</b> (<i>'bool' | 'int' | 'float'</i>) &ndash; state data type
            (<span style="color:#00C000"><b>default</b></span>: 'float').</li>
            <li><b>shape</b> (<i>int | iter[int]</i>) &ndash; state shape
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>num_states</b> (<i>int > 0</i>) &ndash; number of discrete state values
            (<span style="color:#C00000"><b>required</b></span> for type 'int').</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum state value
            (<span style="color:#00C000"><b>optional</b></span> for type 'float').</li>
            </ul>
        actions (specification): Actions specification (**required**), arbitrarily nested
            dictionary of action descriptions (usually taken from `Environment.actions()`) with
            the following attributes:
            <ul>
            <li><b>type</b> (<i>'bool' | 'int' | 'float'</i>) &ndash; action data type
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>shape</b> (<i>int > 0 | iter[int > 0]</i>) &ndash; action shape
            (<span style="color:#00C000"><b>default</b></span>: ()).</li>
            <li><b>num_actions</b> (<i>int > 0</i>) &ndash; number of discrete action values
            (<span style="color:#C00000"><b>required</b></span> for type 'int').</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum action value
            (<span style="color:#00C000"><b>optional</b></span> for type 'float').</li>
            </ul>
        max_episode_timesteps (int > 0): ?
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
            attributes
            (<span style="color:#00C000"><b>default</b></span>: no summarizer):
            <ul>
            <li><b>directory</b> (<i>path</i>) &ndash; summarizer directory
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>steps</b> (<i>int > 0, dict[int > 0]</i>) &ndash; how frequently to record
            summaries, applies to "variables" and "act" if specified globally
            (<span style="color:#00C000"><b>default</b></span>:
            always), otherwise specified per "variables"/"act" in timesteps and "observe"/"update"
            in updates (<span style="color:#00C000"><b>default</b></span>: never).</li>
            <li><b>flush</b> (<i>int > 0</i>) &ndash; how frequently in seconds to flush the
            summary writer (<span style="color:#00C000"><b>default</b></span>: 10).</li>
            <li><b>labels</b> (<i>"all" | iter[string]</i>) &ndash; all or list of summaries to
            record, from the following labels
            (<span style="color:#00C000"><b>default</b></span>: only "graph"):</li>
            <li>"graph": graph summary</li>
            <li>"parameters": parameter scalars</li>
            </ul>
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
