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

import warnings
from tensorforce import TensorForceError
from tensorforce.agents import MemoryAgent
from tensorforce.models import QModel


class DDQNAgent(MemoryAgent):
    """
    Double DQN Agent based on [Van Hasselt et al.](https://arxiv.org/abs/1509.06461). Simple
    extension to DQN which improves stability.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe=1000,
        scope='ddqn',
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
        # parameters specific to double DQN Agent
        target_sync_frequency=10000,
        target_update_weight=1.0,
        huber_loss=None
    ):
        """
        Double Q-learning agent where double-dqn loss is default enabled.

        Args:
            target_sync_frequency: Interval between optimization calls synchronizing the target network.
            target_update_weight: Update weight, 1.0 meaning a full assignment to target network from training network.
            huber_loss: Optional flat specifying Huber-loss clipping.
        """
        # TODO: get rid of this class if we still can. Otherwise, leave this deprecation warning in place
        warnings.warn("WARNING: DDQNAgent is an obsolete class. Instead, use DQNAgent with double_q_model set to True",
                      category=DeprecationWarning)

        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight
        self.huber_loss = huber_loss

        super(DDQNAgent, self).__init__(
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

    def initialize_model(self):
        return QModel(
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
            double_q_model=True,
            huber_loss=self.huber_loss,
            # TEMP: Random sampling fix
            random_sampling_fix=True
        )
