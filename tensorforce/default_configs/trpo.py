# Copyright 2016 reinforce.io. All Rights Reserved.
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
