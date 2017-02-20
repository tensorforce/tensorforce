# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import PGAgent
from tensorforce.models import TRPOModel

from tensorforce.default_configs import TRPOAgentConfig

class TRPOAgent(PGAgent):
    name = 'TRPOAgent'

    model_ref = TRPOModel

    default_config = TRPOAgentConfig
