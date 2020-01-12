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

from collections import OrderedDict

from tensorforce.agents import TensorforceAgent


class PPOBaselines(TensorforceAgent):
    """
    ???
    """






        if update_unit == 'timesteps':
            memory = dict(type='recent', capacity=((batch_size // max_episode_timesteps + 3) * max_episode_timesteps))
        else:



    def __init__(
        # Environment
        self, states, actions, max_episode_timesteps,
        # Network
        network='auto',
        # Optimization
        batch_size=10, update_frequency=None, learning_rate=3e-4, subsampling_fraction=0.33,
        optimization_steps=10,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
        # Critic
        critic_network=None, critic_optimizer=None,
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
        summarizer=None, recorder=None, config=None
    ):
        self.spec = OrderedDict(
            ???
        )

        memory = dict(type='recent', capacity=((batch_size + 1) * max_episode_timesteps))
        if update_frequency is None:
            update = dict(unit=update_unit, batch_size=batch_size)
        else:
            update = dict(unit=update_unit, batch_size=batch_size, frequency=update_frequency)
        optimizer = dict(type='adam', learning_rate=learning_rate)
        optimizer = dict(
            type='subsampling_step', optimizer=optimizer, fraction=subsampling_fraction
        )
        optimizer = dict(type='multi_step', optimizer=optimizer, num_steps=optimization_steps)
        objective = dict(
            type='policy_gradient', ratio_based=True, clipping_value=likelihood_ratio_clipping
        )
        if critic_network is None:
            reward_estimation = dict(horizon='episode', discount=discount)
        else:
            reward_estimation = dict(
                horizon='episode', discount=discount,
                estimate_horizon=('late' if estimate_terminal else False),
                estimate_terminal=estimate_terminal, estimate_advantage=True
            )
        if critic_network is None:
            baseline_policy = None
            baseline_objective = None
        else:
            # State value doesn't exist for Beta
            baseline_policy = dict(network=critic_network, distributions=dict(float='gaussian'))
            assert critic_optimizer is not None
            baseline_objective = 'state_value'

        super().__init__(
            # Agent
            states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
            parallel_interactions=parallel_interactions, buffer_observe=True, seed=seed,
            recorder=recorder, config=config,
            # Model
            name=name, device=device, execution=execution, saver=saver, summarizer=summarizer,
            preprocessing=preprocessing, exploration=exploration, variable_noise=variable_noise,
            l2_regularization=l2_regularization,
            # TensorforceModel
            policy=None, network=network, memory=memory, update=update, optimizer=optimizer,
            objective=objective, reward_estimation=reward_estimation,
            baseline_policy=baseline_policy, baseline_network=None,
            baseline_optimizer=critic_optimizer, baseline_objective=baseline_objective,
            entropy_regularization=entropy_regularization
        )
