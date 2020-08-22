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

from tensorforce.core.policies.base_policy import BasePolicy

from tensorforce.core.policies.action_value import ActionValue
from tensorforce.core.policies.policy import Policy
from tensorforce.core.policies.state_value import StateValue

from tensorforce.core.policies.stochastic_policy import StochasticPolicy
from tensorforce.core.policies.value_policy import ValuePolicy

from tensorforce.core.policies.parametrized_action_value import ParametrizedActionValue
from tensorforce.core.policies.parametrized_distributions import ParametrizedDistributions
from tensorforce.core.policies.parametrized_state_value import ParametrizedStateValue
from tensorforce.core.policies.parametrized_value_policy import ParametrizedValuePolicy


policy_modules = dict(
    parametrized_action_value=ParametrizedActionValue,
    parametrized_distributions=ParametrizedDistributions,
    parametrized_state_value=ParametrizedStateValue,
    parametrized_value_policy=ParametrizedValuePolicy
)


__all__ = [
    'ActionValue', 'BasePolicy', 'ParametrizedActionValue', 'ParametrizedDistributions',
    'ParametrizedStateValue', 'ParametrizedValuePolicy', 'Policy', 'StateValue', 'StochasticPolicy',
    'ValuePolicy'
]
