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

"""
Replay replay_memory testing.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from six.moves import xrange

from tensorforce.replay_memories import ReplayMemory

def test_replay_memory():
    """
    Testing replay replay_memory.
    """
    capacity = np.random.randint(5, 8)
    batch_size = np.random.randint(capacity)

    state_shape = tuple(np.random.randint(1, 4, size=2))
    action_shape = (4,)

    memory = ReplayMemory(capacity, state_shape, action_shape)

    states = []
    actions = []
    rewards = []
    terminals = []

    def sample_observation():
        while True:
            state = np.random.randint(0, 255, size=state_shape)
            if len(states) > 0:
                if not np.all(np.any(np.array(states) - np.array(state), axis=1)):
                    # avoid duplicate states
                    continue
            break

        action = np.random.randint(4)
        reward = np.random.choice(2, 1, p=[0.7, 0.3])
        terminal = np.random.choice(2, 1, p=[0.9, 0.1])

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        terminals.append(terminal)

        memory.add_experience(state, action, reward, terminal)

        return state, action, reward, terminal

    for i in xrange(capacity):
        state, action, reward, terminal = sample_observation()

    assert not np.any(np.array(memory.states) - np.array(states))

    state, action, reward, terminal = sample_observation()

    assert not np.any(np.array(memory.states[0]) - np.array(state))

    for i in xrange(capacity-1):
        state, action, reward, terminal = sample_observation()

    assert not np.any(np.array(memory.states) - np.array(states[-capacity:]))

    batch = memory.sample_batch(batch_size)
    exp = zip(list(batch['states']), batch['actions'], batch['rewards'], batch['terminals'], batch['next_states'])

    # Warning: since we're testing a random batch, some of the following assertions could be True by coincidence
    # In this test, states are unique, so we can just compare state tensors with each other

    for i in xrange(100):
        first_state = states[0]
        last_state = states[-1]
        for (state, action, reward, terminal, next_state) in exp:
            # last state must not be in experiences, as it has no next state
            assert np.all(np.any(state - last_state, axis=1))

            # first state must not be in next_states, as it has no previous state
            assert np.all(np.any(next_state - first_state, axis=1))

