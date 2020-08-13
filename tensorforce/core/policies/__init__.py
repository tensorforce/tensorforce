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

from tensorforce.core.policies.policy import Policy
from tensorforce.core.policies.state_value import StateValue

from tensorforce.core.policies.action_value import ActionValue
from tensorforce.core.policies.stochastic import Stochastic

from tensorforce.core.policies.parametrized_action_value import ParametrizedActionValue
from tensorforce.core.policies.parametrized_distributions import ParametrizedDistributions
from tensorforce.core.policies.parametrized_state_action_value import ParametrizedStateActionValue
from tensorforce.core.policies.parametrized_state_value import ParametrizedStateValue


policy_modules = dict(
    default=ParametrizedDistributions, parametrized_action_value=ParametrizedActionValue,
    parametrized_distributions=ParametrizedDistributions,
    parametrized_state_action_value=ParametrizedStateActionValue,
    parametrized_state_value=ParametrizedStateValue
)


__all__ = [
    'ActionValue', 'ParametrizedActionValue', 'ParametrizedDistributions',
    'ParametrizedStateActionValue', 'ParametrizedStateValue', 'Policy', 'StateValue', 'Stochastic'
]
