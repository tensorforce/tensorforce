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

from tensorforce.agents import Agent


class BatchAgent(Agent):
    """
    The `BatchAgent` class implements a batch memory which generally implies on-policy
    experience collection and updates.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe,
        batch_size,
        keep_last_timestep
    ):
        """

        Args:
            states_spec: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions_spec: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
            batch_size: Int specifying number of samples collected via `observe` before an update is executed.
            keep_last_timestep: Boolean flag specifying whether last sample is kept, default True.
        """
        assert isinstance(batch_size, int) and batch_size > 0
        self.batch_size = batch_size

        assert isinstance(keep_last_timestep, bool)
        self.keep_last_timestep = keep_last_timestep

        super(BatchAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe
        )

        self.batch_count = None
        self.reset_batch()

    def observe(self, terminal, reward):
        """
        Adds an observation and performs an update if the necessary conditions
        are satisfied, i.e. if one batch of experience has been collected as defined
        by the batch size.

        In particular, note that episode control happens outside of the agent since
        the agent should be agnostic to how the training data is created.

        Args:
            reward: float of a scalar reward
            terminal: boolean whether episode is terminated or not

        Returns: void
        """
        super(BatchAgent, self).observe(terminal=terminal, reward=reward)

        for name, batch_state in self.batch_states.items():
            batch_state.append(self.current_states[name])
        for batch_internal, internal in zip(self.batch_internals, self.current_internals):
            batch_internal.append(internal)
        for name, batch_action in self.batch_actions.items():
            batch_action.append(self.current_actions[name])
        self.batch_terminal.append(self.current_terminal)
        self.batch_reward.append(self.current_reward)

        self.batch_count += 1

        if self.batch_count == self.batch_size:
            self.model.update(
                states=self.batch_states,
                internals=self.batch_internals,
                actions=self.batch_actions,
                terminal=self.batch_terminal,
                reward=self.batch_reward
            )
            self.reset_batch()

    def reset_batch(self):
        if self.batch_count is None or not self.keep_last_timestep:
            self.batch_states = {name: list() for name in self.states_spec}
            self.batch_internals = [list() for _ in range(len(self.current_internals))]
            self.batch_actions = {name: list() for name in self.actions_spec}
            self.batch_terminal = list()
            self.batch_reward = list()
            self.batch_count = 0

        else:
            self.batch_states = {name: [self.batch_states[name][-1]] for name in self.states_spec}
            self.batch_internals = [[self.batch_internals[i][-1]] for i in range(len(self.current_internals))]
            self.batch_actions = {name: [self.batch_actions[name][-1]] for name in self.actions_spec}
            self.batch_terminal = [self.batch_terminal[-1]]
            self.batch_reward = [self.batch_reward[-1]]
            self.batch_count = 1
