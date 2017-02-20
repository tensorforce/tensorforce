# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Vanilla policy gradient agent with GAE.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import PGAgent
from tensorforce.models.vpg_model import VPGModel

from tensorforce.default_configs import VPGAgentConfig

class VPGAgent(PGAgent):
    name = 'VPGAgent'

    model_ref = VPGModel

    default_config = VPGAgentConfig
