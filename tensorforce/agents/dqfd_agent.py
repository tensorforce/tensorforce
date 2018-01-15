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
import numpy as np

from tensorforce import TensorForceError
from tensorforce.agents import MemoryAgent
from tensorforce.core.memories import Replay
from tensorforce.models import QDemoModel


class DQFDAgent(MemoryAgent):
    """
    Deep Q-learning from demonstration (DQFD) agent ([Hester et al., 2017](https://arxiv.org/abs/1704.03732)).
    This agent uses DQN to pre-train from demonstration data via an additional supervised loss term.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe=1000,
        scope='dqfd',
        # parameters specific to LearningAgents
        summary_spec=None,
        network_spec=None,
        device=None,
        session_config=None,
        saver_spec=None,
        distributed_spec=None,
        optimizer=None,
        discount=0.99,
        variable_noise=None,
        states_preprocessing_spec=None,
        explorations_spec=None,
        reward_preprocessing_spec=None,
        distributions_spec=None,
        entropy_regularization=None,
        # parameters specific to MemoryAgents
        batch_size=32,
        memory=None,
        first_update=10000,
        update_frequency=4,
        repeat_update=1,
        # parameters specific to DQFD agents
        target_sync_frequency=10000,
        target_update_weight=1.0,
        huber_loss=None,
        expert_margin=0.5,
        supervised_weight=0.1,
        demo_memory_capacity=10000,
        demo_sampling_ratio=0.2
    ):
        """
        Deep Q-learning from demonstration (DQFD) agent ([Hester et al., 2017](https://arxiv.org/abs/1704.03732)).
        This agent uses DQN to pre-train from demonstration data in combination with a supervised loss.

        Args:
            target_sync_frequency: Interval between optimization calls synchronizing the target network.
            target_update_weight: Update weight, 1.0 meaning a full assignment to target network from training network.
            huber_loss: Optional flat specifying Huber-loss clipping.
            expert_margin: Positive float specifying enforced supervised margin between expert action Q-value and other
                Q-values.
            supervised_weight: Weight of supervised loss term.
            demo_memory_capacity: Int describing capacity of expert demonstration memory.
            demo_sampling_ratio: Runtime sampling ratio of expert data.
        """
        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight
        self.huber_loss = huber_loss

        self.expert_margin = expert_margin
        self.supervised_weight = supervised_weight

        super(DQFDAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe,
            scope=scope,
            # parameters specific to LearningAgent
            summary_spec=summary_spec,
            network_spec=network_spec,
            discount=discount,
            device=device,
            session_config=session_config,
            saver_spec=saver_spec,
            distributed_spec=distributed_spec,
            optimizer=optimizer,
            variable_noise=variable_noise,
            states_preprocessing_spec=states_preprocessing_spec,
            explorations_spec=explorations_spec,
            reward_preprocessing_spec=reward_preprocessing_spec,
            distributions_spec=distributions_spec,
            entropy_regularization=entropy_regularization,
            # parameters specific to MemoryAgents
            batch_size=batch_size,
            memory=memory,
            first_update=first_update,
            update_frequency=update_frequency,
            repeat_update=repeat_update
        )

        # The demo_sampling_ratio, called p in paper, controls ratio of expert vs online training samples
        # p = n_demo / (n_demo + n_replay) => n_demo  = p * n_replay / (1 - p)
        self.demo_memory_capacity = demo_memory_capacity
        self.demo_batch_size = int(demo_sampling_ratio * batch_size / (1.0 - demo_sampling_ratio))
        assert self.demo_batch_size > 0, 'Check DQFD sampling parameters to ensure ' \
                                         'demo_batch_size is positive. (Calculated {} based on current' \
                                         ' parameters)'.format(self.demo_batch_size)

        # This is the demonstration memory that we will fill with observations before starting
        # the main training loop
        self.demo_memory = Replay(self.states_spec, self.actions_spec, self.demo_memory_capacity)

    def initialize_model(self):
        return QDemoModel(
            states_spec=self.states_spec,
            actions_spec=self.actions_spec,
            network_spec=self.network_spec,
            device=self.device,
            session_config=self.session_config,
            scope=self.scope,
            saver_spec=self.saver_spec,
            summary_spec=self.summary_spec,
            distributed_spec=self.distributed_spec,
            optimizer=self.optimizer,
            discount=self.discount,
            variable_noise=self.variable_noise,
            states_preprocessing_spec=self.states_preprocessing_spec,
            explorations_spec=self.explorations_spec,
            reward_preprocessing_spec=self.reward_preprocessing_spec,
            distributions_spec=self.distributions_spec,
            entropy_regularization=self.entropy_regularization,
            target_sync_frequency=self.target_sync_frequency,
            target_update_weight=self.target_update_weight,
            # DQFD always uses double dqn, which is a required key for a q-model.
            double_q_model=True,
            huber_loss=self.huber_loss,
            # TEMP: Random sampling fix
            random_sampling_fix=True,
            expert_margin=self.expert_margin,
            supervised_weight=self.supervised_weight
        )

    def observe(self, reward, terminal):
        """
        Adds observations, updates via sampling from memories according to update rate.
        DQFD samples from the online replay memory and the demo memory with
        the fractions controlled by a hyper parameter p called 'expert sampling ratio.
        """
        super(DQFDAgent, self).observe(reward=reward, terminal=terminal)

        if self.timestep >= self.first_update and self.timestep % self.update_frequency == 0:
            for _ in xrange(self.repeat_update):
                batch = self.demo_memory.get_batch(batch_size=self.demo_batch_size, next_states=True)
                self.model.demonstration_update(
                    states={name: np.stack((batch['states'][name],
                                            batch['next_states'][name])) for name in batch['states']},
                    internals=batch['internals'],
                    actions=batch['actions'],
                    terminal=batch['terminal'],
                    reward=batch['reward']
                )

    def import_demonstrations(self, demonstrations):
        """
        Imports demonstrations, i.e. expert observations. Note that for large numbers of observations,
        set_demonstrations is more appropriate, which directly sets memory contents to an array an expects
        a different layout.

        Args:
            demonstrations: List of observation dicts
        """
        for observation in demonstrations:
            if self.unique_state:
                state = dict(state=observation['states'])
            else:
                state = observation['states']
            if self.unique_action:
                action = dict(action=observation['actions'])
            else:
                action = observation['actions']

            self.demo_memory.add_observation(
                states=state,
                internals=observation['internals'],
                actions=action,
                terminal=observation['terminal'],
                reward=observation['reward']
            )

    def set_demonstrations(self, batch):
        """
        Set all demonstrations from batch data. Expects a dict wherein each value contains an array
        containing all states, actions, rewards, terminals and internals respectively.

        Args:
            batch:

        """
        self.demo_memory.set_memory(
            states=batch['states'],
            internals=batch['internals'],
            actions=batch['actions'],
            terminal=batch['terminal'],
            reward=batch['reward']
        )

    def pretrain(self, steps):
        """
        Computes pre-train updates.

        Args:
            steps: Number of updates to execute.

        """
        for _ in xrange(steps):
            # Sample from demo memory.
            batch = self.demo_memory.get_batch(batch_size=self.batch_size, next_states=True)

            # Update using both double Q-learning and supervised double_q_loss.
            self.model.demonstration_update(
                states={name: np.stack((batch['states'][name],
                                        batch['next_states'][name])) for name in batch['states']},
                internals=batch['internals'],
                actions=batch['actions'],
                terminal=batch['terminal'],
                reward=batch['reward']
            )
