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

from tensorforce.core.objectives.objective import Objective

from tensorforce.core.objectives.deterministic_policy_gradient import DeterministicPolicyGradient
from tensorforce.core.objectives.plus import Plus
from tensorforce.core.objectives.policy_gradient import PolicyGradient
from tensorforce.core.objectives.value import Value


objective_modules = dict(
    det_policy_gradient=DeterministicPolicyGradient, plus=Plus, policy_gradient=PolicyGradient,
    value=Value
)


__all__ = [
    'DeterministicPolicyGradient', 'Objective', 'objective_modules', 'Plus', 'PolicyGradient',
    'Value'
]
