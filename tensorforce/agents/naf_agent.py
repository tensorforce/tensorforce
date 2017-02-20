# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================
"""

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import MemoryAgent
from tensorforce.models import NAFModel

from tensorforce.default_configs import NAFAgentConfig

class NAFAgent(MemoryAgent):
    name = 'NAFAgent'

    default_config = NAFAgentConfig

    model_ref = NAFModel
