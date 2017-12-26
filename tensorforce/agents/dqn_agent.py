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

from tensorforce import TensorForceError
from tensorforce.agents import MemoryAgent
from tensorforce.models import QModel


class DQNAgent(MemoryAgent):
    """
    Deep-Q-Network agent (DQN). The piece de resistance of deep reinforcement learning as described by
    [Minh et al. (2015)](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). Includes
    an option for double-DQN (DDQN; [van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461))

    DQN chooses from one of a number of discrete actions by taking the maximum Q-value
    from the value function with one output neuron per available action. DQN uses a replay memory for experience
    playback.

    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        summary_spec=None,
        network_spec=None,
        device=None,
        session_config=None,
        scope='dqn',
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
        target_sync_frequency=10000,
        target_update_weight=1.0,
        double_q_model=False,
        huber_loss=None,
        batched_observe=None,
        batch_size=32,
        memory=None,
        first_update=10000,
        update_frequency=4,
        repeat_update=1
    ):
        """
        Creates a Deep-Q agent.

        Args:
            device: Device string specifying model device.
            session_config: optional tf.ConfigProto with additional desired session configurations
            scope: TensorFlow scope, defaults to agent name (e.g. `dqn`).
            saver_spec: Dict specifying automated saving. Use `directory` to specify where checkpoints are saved. Use
                either `seconds` or `steps` to specify how often the model should be saved. The `load` flag specifies
                if a model is initially loaded (set to True) from a file `file`.
            distributed_spec: Dict specifying distributed functionality. Use `parameter_server` and `replica_model`
                Boolean flags to indicate workers and parameter servers. Use a `cluster_spec` key to pass a TensorFlow
                cluster spec.
            optimizer: Dict specifying optimizer type and its optional parameters, typically a `learning_rate`.
                Available optimizer types include standard TensorFlow optimizers, `natural_gradient`,
                and `evolutionary`. Consult the optimizer test or example configurations for more.
            discount: Float specifying reward discount factor.
            variable_noise: Experimental optional parameter specifying variable noise (NoisyNet).
            states_preprocessing_spec: Optional list of states preprocessors to apply to state  
                (e.g. `image_resize`, `grayscale`).
            explorations_spec: Optional dict specifying action exploration type (epsilon greedy  
                or Gaussian noise).
            reward_preprocessing_spec: Optional dict specifying reward preprocessing.
            distributions_spec: Optional dict specifying action distributions to override default distribution choices.
                Must match action names.
            entropy_regularization: Optional positive float specifying an entropy regularization value.
            target_sync_frequency: Interval between optimization calls synchronizing the target network.
            target_update_weight: Update weight, 1.0 meaning a full assignment to target network from training network.
            huber_loss: Optional flat specifying Huber-loss clipping.
        """

        super(DQNAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            summary_spec=summary_spec,
            network_spec=network_spec,
            batched_observe=batched_observe,
            batch_size=batch_size,
            memory=memory,
            first_update=first_update,
            update_frequency=update_frequency,
            repeat_update=repeat_update
        )

        if optimizer is None:
            self.optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )
        else:
            self.optimizer = optimizer

        self.device = device
        self.session_config = session_config
        self.scope = scope
        self.saver_spec = saver_spec
        self.distributed_spec = distributed_spec
        self.discount = discount
        self.variable_noise = variable_noise
        self.states_preprocessing_spec = states_preprocessing_spec
        self.explorations_spec = explorations_spec
        self.reward_preprocessing_spec = reward_preprocessing_spec
        self.distributions_spec = distributions_spec
        self.entropy_regularization = entropy_regularization
        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight
        self.double_q_model = double_q_model
        self.huber_loss = huber_loss

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
            double_q_model=self.double_q_model,
            huber_loss=self.huber_loss,
            # TEMP: Random sampling fix
            random_sampling_fix=True
        )
