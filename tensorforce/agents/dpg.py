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

from tensorforce import TensorforceError
from tensorforce.agents import PolicyAgent


class DeterministicPolicyGradient(PolicyAgent):
    """
    [???](???) agent
    (specification key: `dpg`).
    """

    def __init__(
        # Environment
        self, states, actions, max_episode_timesteps,
        # Network
        network='auto',
        # Optimization
        memory=10000, batch_size=32, update_frequency=4, start_updating=1000, learning_rate=3e-4,
        # Reward estimation
        n_step=0, discount=0.99,
        # Critic
        critic_network='auto', critic_optimizer='adam',
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
        summarizer=None
    ):
        assert max_episode_timesteps is None or memory >= (batch_size + 1) * max_episode_timesteps
        memory = dict(type='replay', capacity=memory)
        update = dict(
            unit='timesteps', batch_size=batch_size, frequency=update_frequency,
            start=start_updating
        )
        optimizer = dict(type='adam', learning_rate=learning_rate)
        objective = 'dpg'
        reward_estimation = dict(
            horizon=n_step, discount=discount, estimate_horizon='late', estimate_actions=True
        )
        # Action value doesn't exist for Beta
        baseline_policy = dict(network=critic_network, distributions=dict(float='gaussian'))
        baseline_objective = 'action_value'

        super().__init__(
            # Agent
            states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
            parallel_interactions=parallel_interactions, buffer_observe=True, seed=seed,
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

        action_spec = next(iter(self.actions_spec.values()))
        if len(self.actions_spec) > 1 or action_spec['type'] != 'float' or \
                action_spec['shape'] != ():
            raise TensorforceError.unexpected()
