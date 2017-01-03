# Copyright 2016 reinforce.io. All Rights Reserved.
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

"""
Tests for the MemoryAgent and the ReplayMemory.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange

import numpy as np

from tensorforce.rl_agents import MemoryAgent
from tensorforce.updater import Model
from tensorforce.config import create_config


class TestModel(Model):
    def __init__(self, config):
        super(TestModel, self).__init__(config)
        self.config = create_config(config, default={})

        self.actions = self.config.actions

        self.count_updates = 0
        self.count_target_updates = 0

    def get_action(self, state):
        return np.random.randint(0, self.actions)

    def update(self, batch):
        self.count_updates += 1

    def update_target_network(self):
        self.count_target_updates += 1


def test_memoryagent_update_frequency():
    """
    Test MemoryAgent update frequency for SGD and value function updates.

    """
    update_steps = np.random.randint(1, 10)
    target_update_steps = np.random.randint(20, 200)

    state_shape = list(np.random.randint(1, 10, size=2))
    min_replay_size = np.random.randint(int(1e3), int(2e3))

    memory_capacity = np.random.randint(int(5e3), int(1e4))

    config = {
        'actions': np.random.randint(2, 10),
        'batch_size': np.random.randint(2, 32),
        'update_rate': 1.0 / update_steps,
        'target_network_update_rate': 1.0 / target_update_steps,
        'min_replay_size': min_replay_size,
        'deterministic_mode': False,
        'use_target_network': True,
        'memory_capacity': memory_capacity,
        'state_shape': state_shape,
        'action_shape': []
    }

    agent = MemoryAgent(config)
    model = TestModel(config)

    # Set value function manually
    agent.value_function = model

    # Assert config values
    assert agent.batch_size == config['batch_size']
    assert agent.update_steps == update_steps
    assert agent.target_update_steps == target_update_steps
    assert agent.min_replay_size == config['min_replay_size']
    assert agent.use_target_network == config['use_target_network']

    max_steps = np.random.randint(int(1e4), int(2e4))

    print("Testing MemoryAgent for {} steps.".format(max_steps))
    print("Memory capacity: {}".format(config['memory_capacity']))
    print("Min replay size: {}".format(config['min_replay_size']))
    print("Batch size:      {}".format(config['batch_size']))
    print("Update steps:    {}".format(update_steps))
    print("Target steps:    {}".format(target_update_steps))
    print("State shape:     {}".format(state_shape))
    print("Actions:         {}".format(config['actions']))

    print("-" * 16)

    step_count = 0
    history = []
    for step_count in xrange(max_steps):
        while True:
            state = np.random.randint(0, 255, size=state_shape)
            action = agent.get_action(state)
            reward = float(np.random.randint(0, 100) // 80)  # p = .8 for reward = 1
            terminal = bool(np.random.randint(0, 100) // 95)

            # avoid duplicate experiences
            unique = True
            for (hs, ha, hr, tw) in history:
                if hs == state:
                    unique = False
            if unique:
                break

        agent.add_observation(state, action, reward, terminal)
        history.append((state.astype(np.float32), action, reward, terminal))

    # All steps - steps before min_replay_size + possible step if min_replay_size is a step itself

    expected_updates = (step_count + 1) // update_steps\
                       - min_replay_size // update_steps \
                       + int(min_replay_size % update_steps == 0)

    expected_target_updates = (step_count + 1) // target_update_steps \
                              - min_replay_size // target_update_steps \
                              + int(min_replay_size % target_update_steps == 0)

    print("Took {} steps.".format(step_count + 1))
    print("Observed {} updates (expected {})".format(model.count_updates, expected_updates))
    print("Observed {} target updates (expected {})".format(model.count_target_updates, expected_target_updates))
    print("Memory has size {}".format(agent.memory.size))

    assert model.count_updates == expected_updates
    assert model.count_target_updates == expected_target_updates

    assert memory_capacity == agent.memory.size

    batch = agent.memory.sample_batch(config['batch_size'])
    exp = zip(list(batch['states']), batch['actions'], batch['rewards'], batch['terminals'], batch['next_states'])

    # Warning: since we're testing a random batch, some of the following assertions could be True by coincidence
    # In this test, states are unique, so we can just compare state tensors with each other

    first_state = history[0][0]
    last_state = history[-1][0]
    for (state, action, reward, terminal, next_state) in exp:
        # last state must not be in experiences, as it has no next state
        assert np.all(state - last_state)

        # first state must not be in next_states, as it has no previous state
        assert np.all(state - first_state)
