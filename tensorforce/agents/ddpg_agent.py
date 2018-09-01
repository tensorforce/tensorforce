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

from tensorforce.agents import LearningAgent
from tensorforce.models import DPGTargetModel


class DDPGAgent(LearningAgent):
    """
    Deep Deterministic Policy Gradient agent
    ([Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971)).
    """

    def __init__(
        self,
        states,
        actions,
        network,
        critic_network,
        batched_observe=True,
        batching_capacity=1000,
        scope='ddpg',
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
        critic_optimizer=None,
        target_sync_frequency=10000,
        target_update_weight=1.0
    ):
        """
        Initializes the DDPG agent.

        Args:
            update_mode (spec): Update mode specification, with the following attributes:
                - unit: 'timesteps' if given (default: 'timesteps').
                - batch_size: integer (default: 10).
                - frequency: integer (default: batch_size).
            memory (spec): Memory specification, see core.memories module for more information
                (default: {type='replay', include_next_states=true, capacity=1000*batch_size}).
            optimizer (spec): Optimizer specification, see core.optimizers module for more
                information (default: {type='adam', learning_rate=1e-3}).
            critic_network (spec): Critic network specification, size_t0 and size_t1.
            critic_optimizer (spec): Critic optimizer specification, see core.optimizers module for
                more information (default: {type='adam', learning_rate=1e-3}).
            target_sync_frequency (int): Target network sync frequency (default: 10000).
            target_update_weight (float): Target network update weight (default: 1.0).
        """

        # Update mode
        if update_mode is None:
            update_mode = dict(
                unit='timesteps',
                batch_size=10
            )
        elif 'unit' in update_mode:
            assert update_mode['unit'] == 'timesteps'
        else:
            update_mode['unit'] = 'timesteps'

        # Memory
        if memory is None:
            # Assumed episode length of 1000 timesteps.
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

        if critic_optimizer is None:
            critic_optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )

        self.critic_network = critic_network
        self.critic_optimizer = critic_optimizer
        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight

        super(DDPGAgent, self).__init__(
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
        return DPGTargetModel(
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
            critic_network=self.critic_network,
            critic_optimizer=self.critic_optimizer,
            target_sync_frequency=self.target_sync_frequency,
            target_update_weight=self.target_update_weight
        )
