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

from six.moves import xrange

from tensorforce.agents import LearningAgent
from tensorforce.models import QDemoModel


class DQFDAgent(LearningAgent):
    """
    Deep Q-learning from demonstration agent
    ([Hester et al., 2017](https://arxiv.org/abs/1704.03732)).
    """

    def __init__(
        self,
        states,
        actions,
        network,
        batched_observe=True,
        batching_capacity=1000,
        scope='dqfd',
        device=None,
        saver=None,
        summarizer=None,
        execution=None,
        variable_noise=None,
        states_preprocessing=None,
        actions_exploration=None,
        reward_preprocessing=None,
        update_mode=None,
        memory=None,
        optimizer=None,
        discount=0.99,
        distributions=None,
        entropy_regularization=None,
        target_sync_frequency=10000,
        target_update_weight=1.0,
        huber_loss=None,
        # first_update=10000,
        # repeat_update=1
        expert_margin=0.5,
        supervised_weight=0.1,
        demo_memory_capacity=10000,
        demo_sampling_ratio=0.2
    ):
        """
        Initializes the DQFD agent.

        Args:
            update_mode (spec): Update mode specification, with the following attributes:
                - unit: 'timesteps' if given (default: 'timesteps').
                - batch_size: integer (default: 32).
                - frequency: integer (default: 4).
            memory (spec): Memory specification, see core.memories module for more information
                (default: {type='replay', include_next_states=true, capacity=1000*batch_size}).
            optimizer (spec): Optimizer specification, see core.optimizers module for more
                information (default: {type='adam', learning_rate=1e-3}).
            target_sync_frequency (int): Target network sync frequency (default: 10000).
            target_update_weight (float): Target network update weight (default: 1.0).
            huber_loss (float): Huber loss clipping (default: none).
            expert_margin (float): Enforced supervised margin between expert action Q-value and
                other Q-values (default: 0.5).
            supervised_weight (float): Weight of supervised loss term (default: 0.1).
            demo_memory_capacity (int): Capacity of expert demonstration memory (default: 10000).
            demo_sampling_ratio (float): Runtime sampling ratio of expert data (default: 0.2).
        """

        # Update mode
        if update_mode is None:
            update_mode = dict(
                unit='timesteps',
                batch_size=32,
                frequency=4
            )
        elif 'unit' in update_mode:
            assert update_mode['unit'] == 'timesteps'
        else:
            update_mode['unit'] = 'timesteps'

        # Memory
        if memory is None:
            # Default capacity of 1000 batches
            memory = dict(
                type='replay',
                include_next_states=True,
                capacity=(1000 * update_mode['batch_size'])
            )
        else:
            assert memory['include_next_states']

        # Optimizer
        if optimizer is None:
            optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )

        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight
        self.double_q_model = True
        self.huber_loss = huber_loss
        self.expert_margin = expert_margin
        self.supervised_weight = supervised_weight

        self.demo_memory_capacity = demo_memory_capacity
        # The demo_sampling_ratio, called p in paper, controls ratio of expert vs online training samples
        # p = n_demo / (n_demo + n_replay) => n_demo  = p * n_replay / (1 - p)
        self.demo_batch_size = int(demo_sampling_ratio * update_mode['batch_size'] / (1.0 - demo_sampling_ratio))
        assert self.demo_batch_size > 0, 'Check DQFD sampling parameters to ensure ' \
                                         'demo_batch_size is positive. (Calculated {} based on current' \
                                         ' parameters)'.format(self.demo_batch_size)

        # This is the demonstration memory that we will fill with observations before starting
        # the main training loop
        super(DQFDAgent, self).__init__(
            states=states,
            actions=actions,
            batched_observe=batched_observe,
            batching_capacity=batching_capacity,
            scope=scope,
            device=device,
            saver=saver,
            summarizer=summarizer,
            execution=execution,
            variable_noise=variable_noise,
            states_preprocessing=states_preprocessing,
            actions_exploration=actions_exploration,
            reward_preprocessing=reward_preprocessing,
            update_mode=update_mode,
            memory=memory,
            optimizer=optimizer,
            discount=discount,
            network=network,
            distributions=distributions,
            entropy_regularization=entropy_regularization
        )

    def initialize_model(self):
        return QDemoModel(
            states=self.states,
            actions=self.actions,
            scope=self.scope,
            device=self.device,
            saver=self.saver,
            summarizer=self.summarizer,
            execution=self.execution,
            batching_capacity=self.batching_capacity,
            variable_noise=self.variable_noise,
            states_preprocessing=self.states_preprocessing,
            actions_exploration=self.actions_exploration,
            reward_preprocessing=self.reward_preprocessing,
            update_mode=self.update_mode,
            memory=self.memory,
            optimizer=self.optimizer,
            discount=self.discount,
            network=self.network,
            distributions=self.distributions,
            entropy_regularization=self.entropy_regularization,
            target_sync_frequency=self.target_sync_frequency,
            target_update_weight=self.target_update_weight,
            # DQFD always uses double dqn, which is a required key for a q-model.
            double_q_model=True,
            huber_loss=self.huber_loss,
            expert_margin=self.expert_margin,
            supervised_weight=self.supervised_weight,
            demo_memory_capacity=self.demo_memory_capacity,
            demo_batch_size=self.demo_batch_size
        )

    # This is handled by the model now
    # def observe(self, reward, terminal):
    #     """
    #     Adds observations, updates via sampling from memories according to update rate.
    #     DQFD samples from the online replay memory and the demo memory with
    #     the fractions controlled by a hyper parameter p called 'expert sampling ratio.
    #
    #     Args:
    #         reward:
    #         terminal:
    #     """
    #     super(DQFDAgent, self).observe(reward=reward, terminal=terminal)
    #     if self.timestep >= self.first_update and self.timestep % self.update_frequency == 0:
    #         for _ in xrange(self.repeat_update):
    #             self.model.demonstration_update()

    def import_demonstrations(self, demonstrations):
        """
        Imports demonstrations, i.e. expert observations. Note that for large numbers of observations,
        set_demonstrations is more appropriate, which directly sets memory contents to an array an expects
        a different layout.

        Args:
            demonstrations: List of observation dicts
        """
        if isinstance(demonstrations, dict):
            if self.unique_state:
                demonstrations['states'] = dict(state=demonstrations['states'])
            if self.unique_action:
                demonstrations['actions'] = dict(action=demonstrations['actions'])

            self.model.import_demo_experience(**demonstrations)

        else:
            if self.unique_state:
                states = dict(state=list())
            else:
                states = {name: list() for name in demonstrations[0]['states']}
            internals = {name: list() for name in demonstrations[0]['internals']}
            if self.unique_action:
                actions = dict(action=list())
            else:
                actions = {name: list() for name in demonstrations[0]['actions']}
            terminal = list()
            reward = list()

            for demonstration in demonstrations:
                if self.unique_state:
                    states['state'].append(demonstration['states'])
                else:
                    for name, state in states.items():
                        state.append(demonstration['states'][name])
                for name, internal in internals.items():
                    internal.append(demonstration['internals'][name])
                if self.unique_action:
                    actions['action'].append(demonstration['actions'])
                else:
                    for name, action in actions.items():
                        action.append(demonstration['actions'][name])
                terminal.append(demonstration['terminal'])
                reward.append(demonstration['reward'])

            self.model.import_demo_experience(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward
            )

    def pretrain(self, steps):
        """
        Computes pre-train updates.

        Args:
            steps: Number of updates to execute.
        """
        for _ in xrange(steps):
            self.model.demo_update()
