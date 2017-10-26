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
from tensorforce.models import PGProbRatioModel


class PPOAgent(BatchAgent):
    """
    Proximal Policy Optimization agent ([Schulman et al., 2017]
    (https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf).

    ### Configuration options

    #### General:

    * `scope`: TensorFlow variable scope name (default: 'ppo')

    #### Hyperparameters:

    * `batch_size`: Positive integer (**mandatory**)
    * `learning_rate`: positive float (default: 1e-4)
    * `discount`: Positive float, at most 1.0 (default: 0.99)
    * `entropy_regularization`: None or positive float (default: 0.01)
    * `gae_lambda`: None or float between 0.0 and 1.0 (default: none)
    * `normalize_rewards`: Boolean (default: false)
    * `likelihood_ratio_clipping`: None or positive float (default: 0.2)

    #### Multi-step optimizer:

    * `step_optimizer`: Specification dict (default: Adam with learning rate 1e-4)
    * `optimization_steps`: positive integer (default: 10)

    #### Baseline:

    * `baseline_mode`: None or 'states' or 'network' (default: none)
    * `baseline`: None or specification dict, or per-state specification for aggregated baseline (default: none)
    * `baseline_optimizer`: None or specification dict (default: none)

    #### Pre-/post-processing:

    * `state_preprocessing`: None or dict with (default: none)
    * `exploration`: None or dict with (default: none)
    * `reward_preprocessing`: None or dict with (default: none)

    #### Logging:

    * `log_level`: Logging level, one of the following values (default: 'info')
        + 'info', 'debug', 'critical', 'warning', 'fatal'

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

    default_config = dict(
        # Agent
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        # BatchAgent
        keep_last_timestep=True,  # not documented!
        # PPOAgent
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-4
        ),
        optimization_steps=10,
        # Model
        discount=0.99,
        normalize_rewards=False,
        variable_noise=None,  # not documented!!!
        # DistributionModel
        distributions=None,  # not documented!!!
        entropy_regularization=1e-2,
        # PGModel
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None,
        # PGProbRatioModel
        likelihood_ratio_clipping=0.2,
        # Logging
        log_level='info',
        model_directory=None,
        save_frequency=600,  # TensorFlow default
        summary_labels=['total-loss'],
        summary_frequency=120,  # TensorFlow default
        # TensorFlow distributed configuration
        cluster_spec=None,
        parameter_server=False,
        task_index=0,
        device=None,
        local_model=False,
        replica_model=False,
        scope='ppo'
    )

    # missing: batch agent configs
        # entropy_penalty=0.01,
        # loss_clipping=0.2,  # Trust region clipping
        # epochs=10,  # Number of training epochs for SGD,
        # optimizer_batch_size=128,  # Batch size for optimiser
        # random_sampling=True  # Sampling strategy for replay memory

    def __init__(self, states_spec, actions_spec, network_spec, config):
        self.network_spec = network_spec
        config = config.copy()
        config.default(self.__class__.default_config)
        config.obligatory(
            optimizer=dict(
                type='multi_step',
                optimizer=config.step_optimizer,
                num_steps=config.optimization_steps
            )
        )
        super(PPOAgent, self).__init__(states_spec, actions_spec, config)

    def initialize_model(self, states_spec, actions_spec, config):
        return PGProbRatioModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=self.network_spec,
            config=config
        )
