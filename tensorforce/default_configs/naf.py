# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Default configuration for NAF Agent and NAF Model.
"""

NAFAgentConfig = {
    "memory_capacity": 1e5,
    "batch_size": 20,

    "update_rate": 0.25,
    "update_repeat": 1,
    "use_target_network": True,
    "target_network_update_rate": 0.01,
    "min_replay_size": 100
}

NAFModelConfig = {
    "optimizer": "tensorflow.python.training.adam.AdamOptimizer",
    "optimizer_kwargs": {},

    "exploration_mode": "ornstein_uhlenbeck",
    "exploration_param": {
        "sigma": 0.2,
        "mu": 0,
        "theta": 0.15
    },

    "actions": None,

    "alpha": 0.00025,
    "gamma": 0.99,
    "tau": 1.0
}
