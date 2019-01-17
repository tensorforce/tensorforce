# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import unittest

from tensorforce.agents import VPGAgent
from tensorforce.tests.unittest_base import UnittestBase


class TestSummaries(UnittestBase, unittest.TestCase):

    agent = VPGAgent
    config = dict(update_mode=dict(batch_size=2))

    def test_summaries(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        directory = 'summaries-test'

        labels = [
            'bernoulli', 'beta', 'categorical', 'distributions', 'dropout', 'entropy', 'gaussian',
            'graph', 'kl-divergence', 'loss', 'losses', 'objective-loss', 'parameters',
            'regularization-loss', 'relu', 'updates', 'variables'
        ]

        self.unittest(
            name='summaries', states=states, actions=actions, network=network,
            summarizer=dict(directory=directory, labels=labels)
        )

        for filename in os.listdir(path=directory):
            os.remove(path=os.path.join(directory, filename))
        os.rmdir(path=directory)
