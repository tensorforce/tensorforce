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
        states_spec,
        actions_spec,
        device=None,
        scope='random',
        saver_spec=None,
        summary_spec=None,
        distributed_spec=None,
        discount=0.99,
        normalize_rewards=False,
        variable_noise=None,
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        batched_observe=1000
    ):
        """
        Initializes a random agent. Returns random actions based of the shape
        provided in the 'actions_spec'.

        Args:
            states_spec: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions_spec: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            device: Device string specifying model device.
            scope: TensorFlow scope, defaults to agent name (e.g. `dqn`).
            saver_spec: Dict specifying automated saving. Use `directory` to specify where checkpoints are saved. Use
                either `seconds` or `steps` to specify how often the model should be saved. The `load` flag specifies
                if a model is initially loaded (set to True) from a file `file`.
            summary_spec:
            distributed_spec:
            optimizer:
            discount:
            normalize_rewards:
            variable_noise:
            preprocessing: Optional list of preprocessors (e.g. `image_resize`, `grayscale`) to apply to state. Each
                preprocessor is a dict containing a type and optional necessary arguments.
            exploration: Optional dict specifying exploration type (epsilon greedy strategies or Gaussian noise)
                and arguments.
            reward_preprocessing: Optional dict specifying reward preprocessor using same syntax as state preprocessing.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
        """

        self.optimizer = None
        self.device = device
        self.scope = scope
        self.saver_spec = saver_spec
        self.summary_spec = summary_spec
        self.distributed_spec = distributed_spec
        self.discount = discount
        self.normalize_rewards = normalize_rewards
        self.variable_noise = variable_noise


        super(RandomAgent, self).__init__(
            states_spec,
            actions_spec,
            preprocessing=preprocessing,
            exploration=exploration,
            reward_preprocessing=reward_preprocessing,
            batched_observe=batched_observe
         )

    def initialize_model(self, states_spec, actions_spec):
        return RandomModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            device=self.device,
            scope=self.scope,
            saver_spec=self.saver_spec,
            summary_spec=self.summary_spec,
            distributed_spec=self.distributed_spec,
            optimizer=self.optimizer,
            discount=self.discount,
            normalize_rewards=self.normalize_rewards,
            variable_noise=self.variable_noise
        )
