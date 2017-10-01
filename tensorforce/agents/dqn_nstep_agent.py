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

from tensorforce.agents import BatchAgent
from tensorforce.models import QNstepModel


class DQNNstepAgent(BatchAgent):
    """
    Nstep Deep-Q-Network agent (DQN). Uses the last experiences (batch)
    to train on.

    DQN chooses from one of a number of discrete actions by taking the maximum Q-value
    from the value function with one output neuron per available action.

    Configuration:

    Each agent requires the following configuration parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    The `BatchAgent` class additionally requires the following parameters:

    * `batch_size`: integer of the batch size.
    * `keep_last`: bool optionally keep the last observation for use in the next batch

    Each model requires the following configuration parameters:

    * `discount`: float of discount factor (gamma).
    * `learning_rate`: float of learning rate (alpha).
    * `optimizer`: string of optimizer to use (e.g. 'adam').
    * `device`: string of tensorflow device name.
    * `tf_summary`: string directory to write tensorflow summaries. Default None
    * `tf_summary_level`: int indicating which tensorflow summaries to create.
    * `tf_summary_interval`: int number of calls to get_action until writing tensorflow summaries on update.
    * `log_level`: string containing logleve (e.g. 'info').
    * `distributed`: boolean indicating whether to use distributed tensorflow.
    * `global_model`: global model.
    * `session`: session to use.

    The DQN Nstep agent expects the following additional configuration parameters:

    * `target_update_frequency`: int of states between updates of the target network.
    * `update_target_weight`: float of update target weight (tau parameter).
    * `double_dqn`: boolean indicating whether to use double-dqn.
    * `clip_loss`: float if not 0, uses the huber loss with clip_loss as the linear bound




    ### Configuration options

    #### General:

    * `scope`: TensorFlow variable scope name (default: 'vpg')

    #### Hyperparameters:

    * `batch_size`: Positive integer (**mandatory**)
    * `learning_rate`: positive float (default: 1e-3)
    * `discount`: Positive float, at most 1.0 (default: 0.99)
    * `entropy_regularization`: None or positive float (default: none)

    #### Pre-/post-processing:

    * `state_preprocessing`: None or dict with (default: none)
    * `exploration`: None or dict with (default: none)
    * `reward_preprocessing`: None or dict with (default: none)

    #### Logging:

    * `log_level`: Logging level (default: 'info')
        + One of 'info', 'debug', 'critical', 'warning', 'fatal'
    * `tf_summary`: None or dict with the following values (default: none)
        + `logdir`: Directory where TensorFlow event file will be written
        + `level`: TensorFlow summary logging level
            - `0`:
            - `1`:
            - `2`:
            - `3`:
        + `interval`: Number of timesteps between summaries
    """

    default_config = dict(
        # Agent
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        # BatchAgent
        keep_last_timestep=True,  # not documented!
        # DQNAgent
        learning_rate=1e-3,
        # Model
        scope='dqn-nstep',
        discount=0.99,
        # DistributionModel
        distributions=None,  # not documented!!!
        entropy_regularization=None,
        # QModel
        target_update_frequency=10000,  # not documented!!!
        update_target_weight=1.0,  # not documented!!!
        clip_loss=0.0,  # not documented!!!
        double_dqn=False,  # not documented!!!
        # Logging
        log_level='info',
        tf_summary=None
    )

    # missing: memory agent configs

    def __init__(self, states_spec, actions_spec, network_spec, config):
        self.network_spec = network_spec
        config = config.copy()
        config.default(self.__class__.default_config)
        config.obligatory(
            optimizer=dict(
                type='adam',
                learning_rate=config.learning_rate  # or also default?
            )
        )
        super(DQNNstepAgent, self).__init__(states_spec, actions_spec, config)

    def initialize_model(self, states_spec, actions_spec, config):
        return QNstepModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=self.network_spec,
            config=config
        )
