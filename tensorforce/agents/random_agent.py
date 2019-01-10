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

from tensorforce.agents import Agent
from tensorforce.core.models import RandomModel


class RandomAgent(Agent):
    """
    Agent returning random action values.
    """

    def __init__(
        self,
        states,
        actions,
        parallel_interactions=1,
        buffer_observe=1000,
        scope='random',
        device=None,
        saver=None,
        summarizer=None,
        execution=None
    ):
        """
        Initializes the random agent.

        Args:
            scope (str): TensorFlow scope (default: name of agent).
            device: TensorFlow device (default: none)
            saver (spec): Saver specification, with the following attributes (default: none):
                - directory: model directory.
                - file: model filename (optional).
                - seconds or steps: save frequency (default: 600 seconds).
                - load: specifies whether model is loaded, if existent (default: true).
                - basename: optional file basename (default: 'model.ckpt').
            summarizer (spec): Summarizer specification, with the following attributes (default:
                none):
                - directory: summaries directory.
                - seconds or steps: summarize frequency (default: 120 seconds).
                - labels: list of summary labels to record (default: []).
                - meta_param_recorder_class: ???.
            execution (spec): Execution specification (see sanity_check_execution_spec for details).
        """
        super().__init__(
            states=states, actions=actions, parallel_interactions=parallel_interactions,
            buffer_observe=buffer_observe
        )

        self.model = RandomModel(
            # Model
            states=self.states_spec, actions=self.actions_spec, scope=scope, device=device,
            saver=saver, summarizer=summarizer, execution=execution,
            parallel_interactions=self.parallel_interactions, buffer_observe=self.buffer_observe
        )
