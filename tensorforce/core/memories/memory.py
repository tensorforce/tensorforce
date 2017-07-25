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

from tensorforce import util
import tensorforce.core.memories


class Memory(object):

    def __init__(self, capacity, states_config, actions_config):
        self.capacity = capacity
        self.states_config = states_config
        self.actions_config = actions_config

    def add_observation(self, state, action, reward, terminal, internal):
        raise NotImplementedError

    def get_batch(self, batch_size):
        raise NotImplementedError

    def update_batch(self, loss_per_instance):
        raise NotImplementedError

    def set_memory(self, states, actions, rewards, terminals, internals):
        """
        Deletes memory content and sets content to provided observations.

        Args:
            states:
            actions:
            rewards:
            terminals:
            internals:

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def from_config(config, kwargs=None):
        return util.get_object(
            obj=config,
            predefined=tensorforce.core.memories.memories,
            kwargs=kwargs
        )
