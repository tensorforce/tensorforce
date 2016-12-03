# Copyright 2016 reinforce.io. All Rights Reserved.
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
Implements trust region policy optimization with general advantage estimation (TRPO-GAE) as
introduced by Schulman et al.
"""
from tensorforce.updater.value_function import ValueFunction


# Note: Calling this a value function is a little imprecise, since it encapsulates more than a VF.
class TRPOUpdater(ValueFunction):

    def __init__(self, config):
        super(TRPOUpdater, self).__init__(config)

    def get_action(self, state):
            pass

    def update(self, batch):
            pass