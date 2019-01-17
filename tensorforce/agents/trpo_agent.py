# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from tensorforce.agents import DRLAgent
from tensorforce.core.models import PGProbRatioModel


class TRPOAgent(DRLAgent):
    """
    Trust Region Policy Optimization agent
    ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)).
    """

    def __init__(
        self,
        states,
        actions,
        network,
        parallel_interactions=1,
        buffer_observe=1000,
        scope='trpo',
        device=None,
        saver=None,
        summarizer=None,
        execution=None,
        exploration=None,
        variable_noise=None,
        states_preprocessing=None,
        reward_preprocessing=None,
        update_mode=None,
        memory=None,
        discount=None,
        distributions=None,
        entropy_regularization=None,
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None,
        likelihood_ratio_clipping=None,
        learning_rate=1e-3,
        cg_max_iterations=10,
        cg_damping=1e-3,
        cg_unroll_loop=False,
        ls_max_iterations=10,
        ls_accept_ratio=0.9,
        ls_unroll_loop=False
    ):
        """
        Initializes the TRPO agent.

        Args:
            update_mode (spec): Update mode specification, with the following attributes:
                - unit: 'episodes' if given (default: 'episodes').
                - batch_size: integer (default: 10).
                - frequency: integer (default: batch_size).
            memory (spec): Memory specification, see core.memories module for more information
                (default: {type='latest', include_next_states=false, capacity=1000*batch_size}).
            optimizer (spec): TRPO agent implicitly defines a optimized-step natural-gradient
                optimizer.
            baseline_mode (str): One of 'states', 'network' (default: none).
            baseline (spec): Baseline specification, see core.baselines module for more information
                (default: none).
            baseline_optimizer (spec): Baseline optimizer specification, see core.optimizers module
                for more information (default: none).
            gae_lambda (float): Lambda factor for generalized advantage estimation (default: none).
            likelihood_ratio_clipping (float): Likelihood ratio clipping for policy gradient
                (default: none).
            learning_rate (float): Learning rate of natural-gradient optimizer (default: 1e-3).
            cg_max_iterations (int): Conjugate-gradient max iterations (default: 20).
            cg_damping (float): Conjugate-gradient damping (default: 1e-3).
            cg_unroll_loop (bool): Conjugate-gradient unroll loop (default: false).
            ls_max_iterations (int): Line-search max iterations (default: 10).
            ls_accept_ratio (float): Line-search accept ratio (default: 0.9).
            ls_unroll_loop (bool): Line-search unroll loop (default: false).
        """
        super().__init__(
            states=states, actions=actions, parallel_interactions=parallel_interactions,
            buffer_observe=buffer_observe
        )

        # Update mode
        if update_mode is None:
            update_mode = dict(unit='episodes', batch_size=10)
        elif 'unit' in update_mode:
            pass
        else:
            update_mode['unit'] = 'episodes'

        # Memory
        if memory is None:
            # Assumed episode length of 1000 timesteps.
            memory = dict(
                type='latest', include_next_states=False,
                capacity=(1000 * update_mode['batch_size'])
            )
        else:
            assert not memory['include_next_states']

        # if update_mode is `timesteps`, require memory `latest`
        assert (update_mode['unit'] != 'timesteps' or memory['type'] == 'latest')

        # Optimizer
        optimizer = dict(
            type='optimized_step', optimizer=dict(
                type='natural_gradient', learning_rate=learning_rate,
                cg_max_iterations=cg_max_iterations, cg_damping=cg_damping,
                cg_unroll_loop=cg_unroll_loop,
            ), ls_max_iterations=ls_max_iterations, ls_accept_ratio=ls_accept_ratio,
            ls_mode='exponential',  # !!!!!!!!!!!!!
            ls_parameter=0.5,  # !!!!!!!!!!!!!
            ls_unroll_loop=ls_unroll_loop
        )

        self.model = PGProbRatioModel(
            # Model
            states=self.states_spec, actions=self.actions_spec, scope=scope, device=device,
            saver=saver, summarizer=summarizer, execution=execution,
            parallel_interactions=self.parallel_interactions, buffer_observe=self.buffer_observe,
            exploration=exploration, variable_noise=variable_noise,
            states_preprocessing=states_preprocessing, reward_preprocessing=reward_preprocessing,
            # MemoryModel
            update_mode=update_mode, memory=memory, optimizer=optimizer, discount=discount,
            # DistributionModel
            network=network, distributions=distributions,
            entropy_regularization=entropy_regularization,
            # PGModel
            baseline_mode=baseline_mode, baseline=baseline, baseline_optimizer=baseline_optimizer,
            gae_lambda=gae_lambda,
            # PGProbRatioModel
            likelihood_ratio_clipping=likelihood_ratio_clipping
        )
