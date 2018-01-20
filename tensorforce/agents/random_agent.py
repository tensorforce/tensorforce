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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import Agent
from tensorforce.models.random_model import RandomModel


class RandomAgent(Agent):
    """
    Random agent, useful as a baseline and sanity check.
    """

    def __init__(
        self,
        states,
        actions,
        scope='random',
        device=None,
        saver=None,
        summaries=None,
        distributed=None,
        batching_capacity=1000,
        batched_observe=True
    ):
        """
        Initializes a random agent. Returns random actions based of the shape
        provided in the 'actions'.

        Args:
            states: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            device: Device string specifying model device.
            session_config: optional tf.ConfigProto with additional desired session configurations
            scope: TensorFlow scope, defaults to agent name (e.g. `dqn`).
            saver: Dict specifying automated saving. Use `directory` to specify where checkpoints are saved. Use
                either `seconds` or `steps` to specify how often the model should be saved. The `load` flag specifies
                if a model is initially loaded (set to True) from a file `file`.
            summary: Dict specifying summaries for TensorBoard. Requires a 'directory' to store summaries, `steps`
                or `seconds` to specify how often to save summaries, and a list of `labels` to indicate which values
                to export, e.g. `losses`, `variables`. Consult neural network class and model for all available labels.
            distributed: Dict specifying distributed functionality. Use `parameter_server` and `replica_model`
                Boolean flags to indicate workers and parameter servers. Use a `cluster` key to pass a TensorFlow
                cluster spec.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
        """
        self.scope = scope
        self.device = device
        self.saver = saver
        self.summaries = summaries
        self.distributed = distributed
        self.batching_capacity = batching_capacity

        super(RandomAgent, self).__init__(
            states=states,
            actions=actions,
            batched_observe=batched_observe
        )

    def initialize_model(self):
        return RandomModel(
            states=self.states,
            actions=self.actions,
            scope=self.scope,
            device=self.device,
            saver=self.saver,
            summaries=self.summaries,
            distributed=self.distributed,
            batching_capacity=self.batching_capacity
        )
