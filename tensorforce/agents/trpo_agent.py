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
from tensorforce.models import PGProbRatioModel


class TRPOAgent(BatchAgent):
    """
    Trust Region Policy Optimization ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) agent.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe=1000,
        scope='trpo',
        # parameters specific to LearningAgents (except optimizer)
        summary_spec=None,
        network_spec=None,
        device=None,
        session_config=None,
        saver_spec=None,
        distributed_spec=None,
        discount=0.99,
        variable_noise=None,
        states_preprocessing_spec=None,
        explorations_spec=None,
        reward_preprocessing_spec=None,
        distributions_spec=None,
        entropy_regularization=None,
        # parameters specific to BatchAgents
        batch_size=1000,
        keep_last_timestep=True,
        # parameters specific to trust-region pol.opt. Agents
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None,
        likelihood_ratio_clipping=None,
        learning_rate=1e-3,
        cg_max_iterations=20,
        cg_damping=1e-3,
        cg_unroll_loop=False
    ):
        """
        Creates a Trust Region Policy Optimization ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) agent.

        Args:
            baseline_mode: String specifying baseline mode, `states` for a separate baseline per state, `network`
                for sharing parameters with the training network.
            baseline: Optional dict specifying baseline type (e.g. `mlp`, `cnn`), and its layer sizes. Consult
             examples/configs for full example configurations.
            baseline_optimizer: Optional dict specifying an optimizer and its parameters for the baseline
                following the same conventions as the main optimizer.
            gae_lambda: Optional float specifying lambda parameter for generalized advantage estimation.
            likelihood_ratio_clipping: Optional clipping of likelihood ratio between old and new policy.
            learning_rate: Learning rate which may be interpreted differently according to optimizer, e.g. a natural
                gradient optimizer interprets the learning rate as the max kl-divergence between old and updated policy.
            cg_max_iterations: Int > 0 specifying conjugate gradient iterations, typically 10-20 are sufficient to
                find effective approximate solutions.
            cg_damping: Conjugate gradient damping value to increase numerical stability.
            cg_unroll_loop: Boolean indicating whether loop unrolling in TensorFlow is to be used which seems to
                impact performance negatively at this point, default False.
        """

        self.optimizer = dict(
            type='optimized_step',
            optimizer=dict(
                type='natural_gradient',
                learning_rate=learning_rate,
                cg_max_iterations=cg_max_iterations,
                cg_damping=cg_damping,
                cg_unroll_loop=cg_unroll_loop,
            ),
            ls_max_iterations=10,
            ls_accept_ratio=0.9,
            ls_mode='exponential',
            ls_parameter=0.5,
            ls_unroll_loop=False
        )

        self.baseline_mode = baseline_mode
        self.baseline = baseline
        self.baseline_optimizer = baseline_optimizer
        self.gae_lambda = gae_lambda
        self.likelihood_ratio_clipping = likelihood_ratio_clipping

        super(TRPOAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe,
            # parameters specific to LearningAgent
            summary_spec=summary_spec,
            network_spec=network_spec,
            discount=discount,
            device=device,
            session_config=session_config,
            scope=scope,
            saver_spec=saver_spec,
            distributed_spec=distributed_spec,
            optimizer=self.optimizer,  # use our fixed parametrized optimizer
            variable_noise=variable_noise,
            states_preprocessing_spec=states_preprocessing_spec,
            explorations_spec=explorations_spec,
            reward_preprocessing_spec=reward_preprocessing_spec,
            distributions_spec=distributions_spec,
            entropy_regularization=entropy_regularization,
            # parameters specific to BatchAgents
            batch_size=batch_size,
            keep_last_timestep=keep_last_timestep
        )

    def initialize_model(self):
        return PGProbRatioModel(
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
            baseline_mode=self.baseline_mode,
            baseline=self.baseline,
            baseline_optimizer=self.baseline_optimizer,
            gae_lambda=self.gae_lambda,
            likelihood_ratio_clipping=self.likelihood_ratio_clipping
        )
