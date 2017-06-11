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

from tensorforce.core import Agent


class BatchAgent(Agent):

    default_config = dict()

    def __init__(self, config):
        config.default(BatchAgent.default_config)
        super(BatchAgent, self).__init__(config)
        self.batch_size = config.batch_size
        self.batch = None

    def observe(self, state, action, reward, terminal):
        """
        Adds an observation and performs an update if the necessary conditions
        are satisfied, i.e. if one batch of experience has been collected as defined
        by the batch size.

        In particular, note that episode control happens outside of the agent since
        the agent should be agnostic to how the training data is created.

        :param state:
        :param action:
        :param reward:
        :param terminal:
        :return:
        """
        if self.unique_state:
            state = dict(state=state)
        if self.unique_action:
            action = dict(action=action)

        if self.batch is None:
            self.reset_batch()

        for name in self.batch['states']:
            self.batch['states'][name].append(state[name])
        for name in self.batch['actions']:
            self.batch['actions'][name].append(action[name])
        self.batch['rewards'].append(reward)
        self.batch['terminals'].append(terminal)
        for n, internal in enumerate(self.internals):
            self.batch['internals'][n].append(internal)

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
            internals={n: [] for n in range(len(self.internals))}
        )
        self.batch_count = 0
