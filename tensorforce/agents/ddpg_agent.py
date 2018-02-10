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
    Deep Deterministic Policy Gradient agent as described by [Lillicrap et al. (2016)]
    (https://arxiv.org/pdf/1509.02971.pdf).

    """

    def __init__(
        self,
        states,
        actions,
        network,
        batched_observe=True,
        batching_capacity=1000,
        scope='ddpg',
        device=None,
        saver=None,
        summarizer=None,
        distributed=None,
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
        # parameters specific to ddpg agents
        critic_network=None,
        critic_optimizer=None,
        target_sync_frequency=10000,
        target_update_weight=1.0
    ):
        """
        Creates a deep deterministic policy gradient agent.

        Args:
            states: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            network: List of layers specifying a neural network via layer types, sizes and optional arguments
                such as activation or regularisation. Full examples are in the examples/configs folder.
            device: Device string specifying model device.
            scope: TensorFlow scope, defaults to agent name (e.g. `dqn`).
            saver: Dict specifying automated saving. Use `directory` to specify where checkpoints are saved. Use
                either `seconds` or `steps` to specify how often the model should be saved. The `load` flag specifies
                if a model is initially loaded (set to True) from a file `file`.
            distributed: Dict specifying distributed functionality. Use `parameter_server` and `replica_model`
                Boolean flags to indicate workers and parameter servers. Use a `cluster` key to pass a TensorFlow
                cluster spec.
            optimizer: Dict specifying optimizer type and its optional parameters, typically a `learning_rate`.
                Available optimizer types include standard TensorFlow optimizers, `natural_gradient`,
                and `evolutionary`. Consult the optimizer test or example configurations for more.
            discount: Float specifying reward discount factor.
            variable_noise: Experimental optional parameter specifying variable noise (NoisyNet).
            states_preprocessing: Optional list of states preprocessors to apply to state  
                (e.g. `image_resize`, `grayscale`).
            actions_exploration: Optional dict specifying action exploration type (epsilon greedy  
                or Gaussian noise).
            reward_preprocessing: Optional dict specifying reward preprocessing.
            distributions: Optional dict specifying action distributions to override default distribution choices.
                Must match action names.
            entropy_regularization: Optional positive float specifying an entropy regularization value.
            critic_network: Network definition for the critic.
            critic_optimizer: Specs for the critic optimizer.
            target_sync_frequency: Interval between optimization calls synchronizing the target network.
            target_update_weight: Update weight, 1.0 meaning a full assignment to target network from training network.
        """

        # Update mode
        if update_mode is None:
            update_mode = dict(
                unit='timesteps',
                batch_size=10,
                frequency=10
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
            assert memory['type'] == 'replay'

        # Optimizer
        if optimizer is None:
            optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )

        if critic_network is None:
            critic_network = network

        self.critic_network = critic_network

        if critic_optimizer is None:
            critic_optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )

        self.critic_optimizer = critic_optimizer

        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight

        super(DDPGAgent, self).__init__(
            states=states,
            actions=actions,
            network=network,
            batched_observe=batched_observe,
            batching_capacity=batching_capacity,
            scope=scope,
            device=device,
            saver=saver,
            summarizer=summarizer,
            distributed=distributed,
            variable_noise=variable_noise,
            states_preprocessing=states_preprocessing,
            actions_exploration=actions_exploration,
            reward_preprocessing=reward_preprocessing,
            update_mode=update_mode,
            memory=memory,
            optimizer=optimizer,
            discount=discount,
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
            distributed=self.distributed,
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
