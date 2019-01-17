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


class PPOAgent(DRLAgent):
    """
    Proximal Policy Optimization agent ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)).
    """

    def __init__(
        self,
        states,
        actions,
        network,
        parallel_interactions=1,
        buffer_observe=1000,
        scope='ppo',
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
        likelihood_ratio_clipping=0.2,
        step_optimizer=None,
        subsampling_fraction=0.1,
        optimization_steps=50
    ):
        """
        Initializes the PPO agent.

        Args:
            update_mode (spec): Update mode specification, with the following attributes:
                - unit: 'episodes' if given (default: 'episodes').
                - batch_size: integer (default: 10).
                - frequency: integer (default: batch_size).
            memory (spec): Memory specification, see core.memories module for more information
                (default: {type='latest', include_next_states=false, capacity=1000*batch_size}).
            optimizer (spec): PPO agent implicitly defines a multi-step subsampling optimizer.
            baseline_mode (str): One of 'states', 'network' (default: none).
            baseline (spec): Baseline specification, see core.baselines module for more information
                (default: none).
            baseline_optimizer (spec): Baseline optimizer specification, see core.optimizers module
                for more information (default: none).
            gae_lambda (float): Lambda factor for generalized advantage estimation (default: none).
            likelihood_ratio_clipping (float): Likelihood ratio clipping for policy gradient
                (default: 0.2).
            step_optimizer (spec): Step optimizer specification of implicit multi-step subsampling
                optimizer, see core.optimizers module for more information (default: {type='adam',
                learning_rate=1e-3}).
            subsampling_fraction (float): Subsampling fraction of implicit subsampling optimizer
                (default: 0.1).
            optimization_steps (int): Number of optimization steps for implicit multi-step
                optimizer (default: 50).
        """
        super().__init__(
            states=states, actions=actions, parallel_interactions=parallel_interactions,
            buffer_observe=buffer_observe
        )

        # Update mode
        if update_mode is None:
            update_mode = dict(unit='episodes', batch_size=10)
        elif 'unit' in update_mode:
            # assert update_mode['unit'] == 'episodes'
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
        if step_optimizer is None:
            step_optimizer = dict(type='adam', learning_rate=1e-3)
        optimizer = dict(
            type='multi_step', optimizer=dict(
                type='subsampling_step', optimizer=step_optimizer, fraction=subsampling_fraction
            ), num_steps=optimization_steps
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
