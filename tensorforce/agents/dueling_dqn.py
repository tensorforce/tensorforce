# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from tensorforce import TensorforceError
from tensorforce.agents import TensorforceAgent


class DuelingDQN(TensorforceAgent):
    """
    [Dueling DQN](https://arxiv.org/abs/1511.06581) agent (specification key: `dueling_dqn`).

    Args:
        states (specification): States specification
            (<span style="color:#C00000"><b>required</b></span>, better implicitly specified via
            `environment` argument for `Agent.create(...)`), arbitrarily nested dictionary of state
            descriptions (usually taken from `Environment.states()`) with the following attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; state data type
            (<span style="color:#00C000"><b>default</b></span>: "float").</li>
            <li><b>shape</b> (<i>int | iter[int]</i>) &ndash; state shape
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>num_values</b> (<i>int > 0</i>) &ndash; number of discrete state values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum state value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        actions (specification): Actions specification
            (<span style="color:#C00000"><b>required</b></span>, better implicitly specified via
            `environment` argument for `Agent.create(...)`), arbitrarily nested dictionary of
            action descriptions (usually taken from `Environment.actions()`) with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; action data type
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>shape</b> (<i>int > 0 | iter[int > 0]</i>) &ndash; action shape
            (<span style="color:#00C000"><b>default</b></span>: scalar).</li>
            <li><b>num_values</b> (<i>int > 0</i>) &ndash; number of discrete action values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum action value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        max_episode_timesteps (int > 0): Upper bound for numer of timesteps per episode
            (<span style="color:#00C000"><b>default</b></span>: not given, better implicitly
            specified via `environment` argument for `Agent.create(...)`).

        memory (int > 0): Replay memory capacity, has to fit at least maximum batch_size + maximum
            network/estimator horizon + 1 timesteps
            (<span style="color:#C00000"><b>required</b></span>).
        batch_size (<a href="../modules/parameters.html">parameter</a>, int > 0): Number of
            timesteps per update batch
            (<span style="color:#C00000"><b>required</b></span>).

        network ("auto" | specification): Policy network configuration, see the
            [networks documentation](../modules/networks.html)
            (<span style="color:#00C000"><b>default</b></span>: "auto", automatically configured
            network).
        update_frequency ("never" | <a href="../modules/parameters.html">parameter</a>, int > 0):
            Frequency of updates
            (<span style="color:#00C000"><b>default</b></span>: batch_size).
        start_updating (<a href="../modules/parameters.html">parameter</a>, int >= batch_size):
            Number of timesteps before first update
            (<span style="color:#00C000"><b>default</b></span>: none).
        learning_rate (<a href="../modules/parameters.html">parameter</a>, float > 0.0): Optimizer
            learning rate
            (<span style="color:#00C000"><b>default</b></span>: 1e-3).
        huber_loss (<a href="../modules/parameters.html">parameter</a>, float > 0.0): Huber loss
            threshold
            (<span style="color:#00C000"><b>default</b></span>: no huber loss).

        horizon (<a href="../modules/parameters.html">parameter</a>, int >= 1): n-step DQN, horizon
            of discounted-sum reward estimation before target network estimate
            (<span style="color:#00C000"><b>default</b></span>: 1).
        discount (<a href="../modules/parameters.html">parameter</a>, 0.0 <= float <= 1.0): Discount
            factor for future rewards of discounted-sum reward estimation
            (<span style="color:#00C000"><b>default</b></span>: 0.99).
        predict_terminal_values (bool): Whether to predict the value of terminal states
            (<span style="color:#00C000"><b>default</b></span>: false).

        target_update_weight (<a href="../modules/parameters.html">parameter</a>, 0.0 < float <= 1.0):
            Target network update weight
            (<span style="color:#00C000"><b>default</b></span>: 1.0).
        target_sync_frequency (<a href="../modules/parameters.html">parameter</a>, int >= 1):
            Interval between target network updates
            (<span style="color:#00C000"><b>default</b></span>: every update).

        l2_regularization (<a href="../modules/parameters.html">parameter</a>, float >= 0.0):
            L2 regularization loss weight
            (<span style="color:#00C000"><b>default</b></span>: no L2 regularization).
        entropy_regularization (<a href="../modules/parameters.html">parameter</a>, float >= 0.0):
            Entropy regularization loss weight, to discourage the policy distribution from being
            "too certain"
            (<span style="color:#00C000"><b>default</b></span>: no entropy regularization).

        state_preprocessing (dict[specification]): State preprocessing as layer or list of layers,
            see the [preprocessing documentation](../modules/preprocessing.html),
            specified per state-type or -name
            (<span style="color:#00C000"><b>default</b></span>: linear normalization of bounded
            float states to [-2.0, 2.0]).
        reward_preprocessing (specification): Reward preprocessing as layer or list of layers,
            see the [preprocessing documentation](../modules/preprocessing.html)
            (<span style="color:#00C000"><b>default</b></span>: no reward preprocessing).
        exploration (<a href="../modules/parameters.html">parameter</a> | dict[<a href="../modules/parameters.html">parameter</a>], float >= 0.0):
            Exploration, defined as the probability for uniformly random output in case of `bool`
            and `int` actions, and the standard deviation of Gaussian noise added to every output in
            case of `float` actions, specified globally or per action-type or -name
            (<span style="color:#00C000"><b>default</b></span>: no exploration).
        variable_noise (<a href="../modules/parameters.html">parameter</a>, float >= 0.0):
            Add Gaussian noise with given standard deviation to all trainable variables, as
            alternative exploration mechanism
            (<span style="color:#00C000"><b>default</b></span>: no variable noise).

        others: See the [Tensorforce agent documentation](tensorforce.html).
    """

    def __init__(
        # Required
        self, states, actions, memory, batch_size,
        # Environment
        max_episode_timesteps=None,
        # Network
        network='auto',
        # Optimization
        update_frequency='batch_size', start_updating=None, learning_rate=1e-3, huber_loss=None,
        # Reward estimation
        horizon=1, discount=0.99, predict_terminal_values=False,
        # Target network
        target_update_weight=1.0, target_sync_frequency=1,
        # Preprocessing
        state_preprocessing='linear_normalization', reward_preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # Parallel interactions
        parallel_interactions=1,
        # Config, saver, summarizer, recorder
        config=None, saver=None, summarizer=None, recorder=None,
        # Deprecated
        estimate_terminal=None, **kwargs
    ):
        if estimate_terminal is not None:
            raise TensorforceError.deprecated(
                name='DuelingDQN', argument='estimate_terminal',
                replacement='predict_terminal_values'
            )

        self.spec = OrderedDict(
            agent='dueling_dqn',
            states=states, actions=actions, memory=memory, batch_size=batch_size,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            update_frequency=update_frequency, start_updating=start_updating,
            learning_rate=learning_rate, huber_loss=huber_loss,
            horizon=horizon, discount=discount, predict_terminal_values=predict_terminal_values,
            target_update_weight=target_update_weight, target_sync_frequency=target_sync_frequency,
            state_preprocessing=state_preprocessing, reward_preprocessing=reward_preprocessing,
            exploration=exploration, variable_noise=variable_noise,
            l2_regularization=l2_regularization, entropy_regularization=entropy_regularization,
            parallel_interactions=parallel_interactions,
            config=config, saver=saver, summarizer=summarizer, recorder=recorder
        )

        policy = dict(type='parametrized_value_policy', network=network)

        memory = dict(type='replay', capacity=memory)

        update = dict(unit='timesteps', batch_size=batch_size)
        if update_frequency != 'batch_size':
            update['frequency'] = update_frequency
        if start_updating is not None:
            update['start'] = start_updating

        optimizer = dict(type='adam', learning_rate=learning_rate)
        objective = dict(type='value', value='action', huber_loss=huber_loss)

        reward_estimation = dict(
            horizon=horizon, discount=discount, predict_horizon_values='late',
            estimate_advantage=False, predict_action_values=True,
            predict_terminal_values=predict_terminal_values
        )

        baseline = policy
        baseline_optimizer = dict(
            type='synchronization', update_weight=target_update_weight,
            sync_frequency=target_sync_frequency
        )
        baseline_objective = None

        super().__init__(
            # Agent
            states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
            parallel_interactions=parallel_interactions, config=config, recorder=recorder,
            # TensorforceModel
            policy=policy, memory=memory, update=update, optimizer=optimizer, objective=objective,
            reward_estimation=reward_estimation,
            baseline=baseline, baseline_optimizer=baseline_optimizer,
            baseline_objective=baseline_objective,
            l2_regularization=l2_regularization, entropy_regularization=entropy_regularization,
            state_preprocessing=state_preprocessing, reward_preprocessing=reward_preprocessing,
            exploration=exploration, variable_noise=variable_noise,
            saver=saver, summarizer=summarizer, **kwargs
        )

        if any(spec.type != 'int' for spec in self.actions_spec.values()):
            raise TensorforceError.value(
                name='DuelingDQN', argument='actions', value=actions, hint='contains non-int action'
            )
