# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Default configuration for NAF Agent and VPG Model.
"""

VPGAgentConfig = {
    "batch_size": 1000,
}

VPGModelConfig = {
    "optimizer": "tensorflow.python.training.adam.AdamOptimizer",
    "optimizer_kwargs": {},

    "actions": None,
    "continuous": False,

    "alpha": 0.00025,
    "gamma": 0.97,
    "use_gae": False,
    "gae_gamma": 0.97,

    "normalize_advantage": False
}
