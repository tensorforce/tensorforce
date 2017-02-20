# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================


from tensorforce.default_configs.dqn import DQNAgentConfig, DQNModelConfig
from tensorforce.default_configs.naf import NAFAgentConfig, NAFModelConfig
from tensorforce.default_configs.vpg import VPGAgentConfig, VPGModelConfig
from tensorforce.default_configs.trpo import TRPOAgentConfig, TRPOModelConfig

__all__ = [
    'DQNAgentConfig', 'DQNModelConfig',
    'NAFAgentConfig', 'NAFModelConfig',
    'VPGAgentConfig', 'VPGModelConfig',
    'TRPOAgentConfig', 'TRPOModelConfig'
]
