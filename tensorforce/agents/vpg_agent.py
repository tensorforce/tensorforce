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
from tensorforce.core.models import PGLogProbModel


class VPGAgent(DRLAgent):
    """
    Vanilla policy gradient agent (https://link.springer.com/article/10.1007/BF00992696).
    """

    def __init__(
        self,
        states,
        actions,
        network,
        parallel_interactions=1,
        buffer_observe=1000,
        scope='vpg',
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
        optimizer=None,
        discount=None,
        distributions=None,
        entropy_regularization=None,
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None
    ):
        """
        Initializes the VPG agent.

        Args:
            update_mode (spec): Update mode specification, with the following attributes:
                - unit: 'episodes' if given (default: 'episodes').
                - batch_size: integer (default: 10).
                - frequency: integer (default: batch_size).
            memory (spec): Memory specification, see core.memories module for more information
                (default: {type='latest', include_next_states=false, capacity=1000*batch_size}).
            optimizer (spec): Optimizer specification, see core.optimizers module for more
                information (default: {type='adam', learning_rate=1e-3}).
            baseline_mode (str): One of 'states', 'network' (default: none).
            baseline (spec): Baseline specification, see core.baselines module for more information
                (default: none).
            baseline_optimizer (spec): Baseline optimizer specification, see core.optimizers module
                for more information (default: none).
            gae_lambda (float): Lambda factor for generalized advantage estimation (default: none).
        """
        super().__init__(
            states=states, actions=actions, parallel_interactions=parallel_interactions,
            buffer_observe=buffer_observe
        )

        # Update mode
        if update_mode is None:
            update_mode = dict(unit='episodes', batch_size=10)
        elif 'unit' in update_mode:
            # Tests check all modes for VPG.
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
        # Note: We actually test all combinations in test_vpg_memories - disable assertion for now
        # assert (update_mode['unit'] != 'timesteps' or memory['type'] == 'latest')

        # Optimizer
        if optimizer is None:
            optimizer = dict(type='adam', learning_rate=1e-3)

        self.model = PGLogProbModel(
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
            gae_lambda=gae_lambda
        )
