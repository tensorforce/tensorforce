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

from tensorforce.agents import PolicyAgent


class ActorCritic(PolicyAgent):
    """
    [Actor-Critic](???) agent
    (specification key: `ac`).
    """

    def __init__(
        # Environment
        self, states, actions, max_episode_timesteps,
        # Network
        network='auto',
        # Optimization
        batch_size=10, update_frequency=None, learning_rate=3e-4,
        # Reward estimation
        horizon=0, discount=0.99, state_action_value=False, estimate_terminal=False,
        # Critic
        critic_network='auto', critic_optimizer=1.0,
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
        memory = dict(type='recent', capacity=(batch_size + max_episode_timesteps + horizon))
        if update_frequency is None:
            update = dict(unit='timesteps', batch_size=batch_size)
        else:
            update = dict(unit='timesteps', batch_size=batch_size, frequency=update_frequency)
        optimizer = dict(type='adam', learning_rate=learning_rate)
        objective = 'policy_gradient'
        reward_estimation = dict(
            horizon=horizon, discount=discount, estimate_horizon='late',
            estimate_actions=state_action_value, estimate_terminal=estimate_terminal
        )
        # State value doesn't exist for Beta
        baseline_policy = dict(network=critic_network, distributions=dict(float='gaussian'))
        if state_action_value:
            baseline_objective = 'action_value'
        else:
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
            # PolicyModel
            policy=None, network=network, memory=memory, update=update, optimizer=optimizer,
            objective=objective, reward_estimation=reward_estimation,
            baseline_policy=baseline_policy, baseline_network=None,
            baseline_optimizer=critic_optimizer, baseline_objective=baseline_objective,
            entropy_regularization=entropy_regularization
        )
