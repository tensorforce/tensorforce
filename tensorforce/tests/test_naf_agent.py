from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest

from tensorforce import Configuration
from tensorforce.agents import NAFAgent
from tensorforce.tests.base_agent_test import BaseAgentTest


class TestNAFAgent(BaseAgentTest, unittest.TestCase):

    agent = NAFAgent
    deterministic = True

    config = Configuration(
        batch_size=8,
        memory_capacity=800,
        first_update=80,
        target_update_frequency=20
    )

    exclude_bool = True
    exclude_int = True
    exclude_bounded = True
    exclude_multi = True
    exclude_lstm = True
