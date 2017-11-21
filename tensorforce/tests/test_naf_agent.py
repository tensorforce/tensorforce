from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest

from tensorforce.agents import NAFAgent
from tensorforce.tests.base_agent_test import BaseAgentTest


class TestNAFAgent(BaseAgentTest, unittest.TestCase):

    agent = NAFAgent
    deterministic = True

    kwargs = dict(
        memory=dict(
            type='replay',
            capacity=1000
        ),
        optimizer=dict(
            type='adam',
            learning_rate=0.001
        ),
        repeat_update=4,
        batch_size=64,
        first_update=64,
        target_sync_frequency=10
    )

    exclude_bool = True
    exclude_int = True
    exclude_bounded = True
    exclude_multi = True
    exclude_lstm = True
