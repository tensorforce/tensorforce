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

"""
Manages batching and episodes internally.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import Agent


class BatchAgent(Agent):
    """
    The `BatchAgent` class implements a batch memory, which is cleared after every update.

    Each agent requires the following ``Configuration`` parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    The `BatchAgent` class additionally requires the following parameters:

    * `batch_size`: integer of the batch size.

    """

    default_config = dict(
        batch_size=1
    )

    def __init__(self, config, model=None):
        config.default(BatchAgent.default_config)
        super(BatchAgent, self).__init__(config, model)
        self.batch_size = config.batch_size
        self.batch = None

    def observe(self, reward, terminal):
        """
        Adds an observation and performs an update if the necessary conditions
        are satisfied, i.e. if one batch of experience has been collected as defined
        by the batch size.

        In particular, note that episode control happens outside of the agent since
        the agent should be agnostic to how the training data is created.

        Args:
            reward: float of a scalar reward
            terminal: boolean whether episode is terminated or not

        Returns: void
        """
        self.current_reward = reward
        self.current_terminal = terminal

        if self.batch is None:
            self.reset_batch()
        for name, batch_state in self.batch['states'].items():
            batch_state.append(self.current_state[name])
        for name, batch_action in self.batch['actions'].items():
            batch_action.append(self.current_action[name])
        self.batch['rewards'].append(self.current_reward)
        self.batch['terminals'].append(self.current_terminal)
        for batch_internal, internal in zip(self.batch['internals'], self.current_internal):
            batch_internal.append(internal)

        self.batch_count += 1
        if self.batch_count == self.batch_size:
            self.model.update(self.batch)
            self.reset_batch()

    def reset_batch(self):
        self.batch = dict(
            states={state: [] for state, _ in self.states_config},
            actions={action: [] for action, _ in self.actions_config},
            rewards=[],
            terminals=[],
            internals=[[] for _ in range(len(self.current_internal))]
        )
        self.batch_count = 0
