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
from tensorforce.models import QNAFModel


class NAFAgent(MemoryAgent):
    """
    NAF: https://arxiv.org/abs/1603.00748

    ### Configuration options

    #### General:

    * `scope`: TensorFlow variable scope name (default: 'vpg')

    #### Hyperparameters:

    * `batch_size`: Positive integer (**mandatory**)
    * `learning_rate`: positive float (default: 1e-3)
    * `discount`: Positive float, at most 1.0 (default: 0.99)
    * `normalize_rewards`: Boolean (default: false)
    * `entropy_regularization`: None or positive float (default: none)

    #### Optimizer:

    * `optimizer`: Specification dict (default: Adam with learning rate 1e-3)

    #### Pre-/post-processing:

    * `state_preprocessing`: None or dict with (default: none)
    * `exploration`: None or dict with (default: none)
    * `reward_preprocessing`: None or dict with (default: none)

    #### TensorFlow Summaries:
    * `summary_logdir`: None or summary directory string (default: none)
    * `summary_labels`: List of summary labels to be reported, some possible values below (default: 'total-loss')
        + 'total-loss'
        + 'losses'
        + 'variables'
        + 'activations'
        + 'relu'
    * `summary_frequency`: Positive integer (default: 1)
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        network_spec,
        device=None,
        scope='naf',
        saver_spec=None,
        summary_spec=None,
        distributed_spec=None,
        optimizer=None,
        discount=0.99,
        normalize_rewards=False,
        variable_noise=None,
        distributions_spec=None,
        entropy_regularization=None,
        target_sync_frequency=10000,
        target_update_weight=1.0,
        double_q_model=False,
        huber_loss=None,
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        batched_observe=1000,
        batch_size=32,
        memory=None,
        first_update=10000,
        update_frequency=4,
        repeat_update=1
    ):
        """
        Creates a NAF-agent which is DQN-variant for continuous actions:
        https://arxiv.org/abs/1603.00748

        Args:
            states_spec:
            actions_spec:
            network_spec:
            device:
            scope:
            saver_spec:
            summary_spec:
            distributed_spec:
            optimizer:
            discount:
            normalize_rewards:
            variable_noise:
            distributions_spec:
            entropy_regularization:
            target_sync_frequency:
            target_update_weight:
            double_q_model:
            huber_loss:
            preprocessing:
            exploration:
            reward_preprocessing:
            batched_observe:
            batch_size:
            memory:
            first_update:
            update_frequency:
            repeat_update:
        """
        if network_spec is None:
            raise TensorForceError("No network_spec provided.")

        if optimizer is None:
            self.optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )
        else:
            self.optimizer = optimizer
        if memory is None:
            memory = dict(
                type='replay',
                capacity=100000
            )
        else:
            self.memory = memory

        self.network_spec = network_spec
        self.device = device
        self.scope = scope
        self.saver_spec = saver_spec
        self.summary_spec = summary_spec
        self.distributed_spec = distributed_spec
        self.discount = discount
        self.normalize_rewards = normalize_rewards
        self.variable_noise = variable_noise
        self.distributions_spec = distributions_spec
        self.entropy_regularization = entropy_regularization
        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight
        self.double_q_model = double_q_model
        self.huber_loss = huber_loss

        super(NAFAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            preprocessing=preprocessing,
            exploration=exploration,
            reward_preprocessing=reward_preprocessing,
            batched_observe=batched_observe,
            batch_size=batch_size,
            memory=memory,
            first_update=first_update,
            update_frequency=update_frequency,
            repeat_update=repeat_update
        )

    def initialize_model(self, states_spec, actions_spec):
        return QNAFModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=self.network_spec,
            device=self.device,
            scope=self.scope,
            saver_spec=self.saver_spec,
            summary_spec=self.summary_spec,
            distributed_spec=self.distributed_spec,
            optimizer=self.optimizer,
            discount=self.discount,
            normalize_rewards=self.normalize_rewards,
            variable_noise=self.variable_noise,
            distributions_spec=self.distributions_spec,
            entropy_regularization=self.entropy_regularization,
            target_sync_frequency=self.target_sync_frequency,
            target_update_weight=self.target_update_weight,
            double_q_model=self.double_q_model,
            huber_loss=self.huber_loss,
            # TEMP: Random sampling fix
            random_sampling_fix=True
        )
