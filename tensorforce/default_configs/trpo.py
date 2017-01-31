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
Default configuration for TRPO Agent and TRPO Model.
"""

TRPOAgentConfig = {
    "memory_capacity": 1e5,
    "batch_size": 20,

    "update_rate": 0.25,
    "update_repeat": 1,
    "use_target_network": True,
    "target_network_update_rate": 0.01,
    "min_replay_size": 100
}

TRPOModelConfig = {
    "optimizer": "tensorflow.python.training.adam.AdamOptimizer",
    "optimizer_kwargs": {},

    "exploration_mode": "ornstein_uhlenbeck",
    "exploration_param": {
        "sigma": 0.2,
        "mu": 0,
        "theta": 0.15
    },

    "actions": None,
    "continuous": False,

    "alpha": 0.00025,
    "gamma": 0.97,
    "use_gae": False,
    "gae_gamma": 0.97,

    "cg_iterations": 20,
    "cg_camping": 0.001,
    "line_search_steps": 20,
    "max_kl_divergence": 0.001,

    "normalize_advantage": False
}
