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
Default configuration for TRPO Agent and TRPO Model.
"""

TRPOAgentConfig = {
    "batch_size": 1000,

}

TRPOModelConfig = {
    "actions": None,
    "continuous": False,

    "gamma": 0.97,
    "use_gae": False,
    "gae_gamma": 0.97,

    "cg_iterations": 20,
    "cg_damping": 0.001,
    "line_search_steps": 20,
    "max_kl_divergence": 0.001,

    "normalize_advantage": False
}
