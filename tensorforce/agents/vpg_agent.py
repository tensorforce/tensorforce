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
from tensorforce.agents import BatchAgent
from tensorforce.models import PGLogProbModel


class VPGAgent(BatchAgent):
    """
    Vanilla Policy Gradient agent as described by [Sutton et al. (1999)]
    (https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf).

    ### Configuration options

    #### General:

    * `scope`: TensorFlow variable scope name (default: 'vpg')

    #### Hyperparameters:

    * `batch_size`: Positive integer (**mandatory**)
    * `discount`: Positive float, at most 1.0 (default: 0.99)
    * `normalize_rewards`: Boolean (default: false)
    * `entropy_regularization`: None or positive float (default: none)
    * `gae_lambda`: None or float between 0.0 and 1.0 (default: none)

    #### Optimizer:

    * `optimizer`: Specification dict (default: Adam with learning rate 1e-3)

    #### Baseline:

    * `baseline_mode`: None, or one of 'states' or 'network' specifying the baseline input (default: none)
    * `baseline`: None or specification dict, or per-state specification for aggregated baseline (default: none)
    * `baseline_optimizer`: None or specification dict (default: none)

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
        scope='vpg',
        saver_spec=None,
        summary_spec=None,
        distributed_spec=None,
        optimizer=None,
        discount=0.99,
        normalize_rewards=False,
        variable_noise=None,
        distributions_spec=None,
        entropy_regularization=None,
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None,
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        batched_observe=1000,
        batch_size=1000,
        keep_last_timestep=True
    ):
        """

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
            baseline_mode:
            baseline:
            baseline_optimizer:
            gae_lambda:
            preprocessing:
            exploration:
            reward_preprocessing:
            batched_observe:
            batch_size:
            keep_last_timestep:
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

        # Model arguments
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
        self.baseline_mode = baseline_mode
        self.baseline = baseline
        self.baseline_optimizer = baseline_optimizer
        self.gae_lambda = gae_lambda

        super(VPGAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            preprocessing=preprocessing,
            exploration=exploration,
            reward_preprocessing=reward_preprocessing,
            batched_observe=batched_observe,
            batch_size=batch_size,
            keep_last_timestep=keep_last_timestep
        )

    def initialize_model(self, states_spec, actions_spec):
        return PGLogProbModel(
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
            baseline_mode=self.baseline_mode,
            baseline=self.baseline,
            baseline_optimizer=self.baseline_optimizer,
            gae_lambda=self.gae_lambda
        )
