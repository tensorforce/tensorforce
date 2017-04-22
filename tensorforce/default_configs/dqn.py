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
Default configuration for DQN Agent and DQN Model.
"""

DQNAgentConfig = {
    "loglevel": "debug",
    "memory_capacity": 1e5,
    "batch_size": 32,

    "update_rate": 0.25,
    "update_repeat": 1,
    "use_target_network": True,
    "target_network_update_rate": 0.0001,
    "min_replay_size": 5e4
}

DQNModelConfig = {
    "optimizer": "tensorflow.python.training.rmsprop.RMSPropOptimizer",
    "optimizer_kwargs": {
        "momentum": 0.95,
        "epsilon": 0.01
    },

    "exploration": "epsilon_decay",
    "exploration_kwargs": {
        "epsilon": 1.0,
        "epsilon_final": 0.1,
        "epsilon_states": 1e6
    },

    "actions": None,

    "alpha": 0.00025,
    "gamma": 0.99,
    "tau": 1.0,

    "double_dqn": False,

    "clip_gradients": True,
    "clip_value": 1.0
}
