# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest

from tensorforce.tests.base_agent_test import BaseAgentTest
from tensorforce.agents import DDPGAgent


class TestDDPGAgent(BaseAgentTest, unittest.TestCase):

    critic_spec = [
        [

            dict(type='input', names=["state"]),
            dict(type="linear", size=64),
            dict(
                type="tf_layer",
                layer="batch_normalization",
                center=True,
                scale=True
            ),
            dict(
                type="nonlinearity",
                name="relu"
            ),
            dict(
                type="output",
                name="state_output"
            )
        ],
        [
            dict(
                type="input",
                names=["state_output", "action"],
                aggregation_type= "concat"
            ),
            dict(
                type="linear",
                size=64
            ),
            dict(
                type="tf_layer",
                layer="batch_normalization",
                center=True,
                scale=True
            ),
            dict(
                type="nonlinearity",
                name="relu"
            ),
            dict(
                type="dense",
                activation="tanh",
                size=1
            )
        ]
    ]

    agent = DDPGAgent
    config = dict(
        update_mode=dict(
            unit='timesteps',
            batch_size=8,
            frequency=8
        ),
        memory=dict(
            type='replay',
            include_next_states=True,
            capacity=100
        ),
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        critic_network=critic_spec,
        target_sync_frequency=10
    )
    exclude_multi = True
