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
Default configuration for NAF Agent and VPG Model.
"""

VPGAgentConfig = {
    "batch_size": 100,
    "continuous": False,
}

VPGModelConfig = {
    "optimizer": "tensorflow.python.training.adam.AdamOptimizer",
    "optimizer_kwargs": {},

    "actions": None,

    "alpha": 0.01,
    "gamma": 0.97,
    "use_gae": False,
    "gae_gamma": 0.97,

    "normalize_advantage": False
}
