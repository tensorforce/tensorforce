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

        network ("auto" | specification): Policy network configuration, see
            [networks](../modules/networks.html)
            (<span style="color:#00C000"><b>default</b></span>: "auto", automatically configured
            network).

        memory (int): Replay memory capacity
            (<span style="color:#00C000"><b>default</b></span>: 10000).
        batch_size (parameter, long > 0): Number of timesteps per update batch
            (<span style="color:#00C000"><b>default</b></span>: 32 timesteps).
        update_frequency ("never" | parameter, long > 0): Frequency of updates
            (<span style="color:#00C000"><b>default</b></span>: every 4 timesteps).
        start_updating (parameter, long >= batch_size): Number of timesteps before first update
            (<span style="color:#00C000"><b>default</b></span>: none).
        learning_rate (parameter, float > 0.0): Optimizer learning rate
            (<span style="color:#00C000"><b>default</b></span>: 3e-4).
        huber_loss (parameter, float > 0.0): Huber loss threshold
            (<span style="color:#00C000"><b>default</b></span>: no huber loss).

        horizon ("episode" | parameter, long >= 0): Horizon of discounted-sum reward estimation
            before critic estimate
            (<span style="color:#00C000"><b>default</b></span>: 0).
        discount (parameter, 0.0 <= float <= 1.0): Discount factor for future rewards of
            discounted-sum reward estimation
            (<span style="color:#00C000"><b>default</b></span>: 0.99).
        estimate_terminal (bool): Whether to estimate the value of (real) terminal states
            (<span style="color:#00C000"><b>default</b></span>: false).

        target_sync_frequency (parameter, int > 0): Interval between target network updates
            (<span style="color:#00C000"><b>default</b></span>: every update).
        target_update_weight (parameter, 0.0 < float <= 1.0): Target network update weight
            (<span style="color:#00C000"><b>default</b></span>: 1.0).

        preprocessing (dict[specification]): Preprocessing as layer or list of layers, see
            [preprocessing](../modules/preprocessing.html), specified per state-type or -name and
            for reward
            (<span style="color:#00C000"><b>default</b></span>: none).

        exploration (parameter | dict[parameter], float >= 0.0): Exploration, global or per action,
            defined as the probability for uniformly random output in case of `bool` and `int`
            actions, and the standard deviation of Gaussian noise added to every output in case of
            `float` actions (<span style="color:#00C000"><b>default</b></span>: 0.0).
        variable_noise (parameter, float >= 0.0): Standard deviation of Gaussian noise added to all
            trainable float variables (<span style="color:#00C000"><b>default</b></span>: 0.0).

        l2_regularization (parameter, float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>:
            0.0).
        entropy_regularization (parameter, float >= 0.0): Scalar controlling entropy
            regularization, to discourage the policy distribution being too "certain" / spiked
            (<span style="color:#00C000"><b>default</b></span>: 0.0).

        name (string): Agent name, used e.g. for TensorFlow scopes
            (<span style="color:#00C000"><b>default</b></span>: "agent").
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: TensorFlow default).
        parallel_interactions (int > 0): Maximum number of parallel interactions to support,
            for instance, to enable multiple parallel episodes, environments or (centrally
            controlled) agents within an environment
            (<span style="color:#00C000"><b>default</b></span>: 1).
        seed (int): Random seed to set for Python, NumPy (both set globally!) and TensorFlow,
            environment seed has to be set separately for a fully deterministic execution
            (<span style="color:#00C000"><b>default</b></span>: none).
        execution (specification): TensorFlow execution configuration with the following attributes
            (<span style="color:#00C000"><b>default</b></span>: standard): ...
        saver (specification): TensorFlow saver configuration with the following attributes
            (<span style="color:#00C000"><b>default</b></span>: no saver):
            <ul>
            <li><b>directory</b> (<i>path</i>) &ndash; saver directory
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>filename</b> (<i>string</i>) &ndash; model filename
            (<span style="color:#00C000"><b>default</b></span>: "agent").</li>
            <li><b>frequency</b> (<i>int > 0</i>) &ndash; how frequently in seconds to save the
            model (<span style="color:#00C000"><b>default</b></span>: 600 seconds).</li>
            <li><b>load</b> (<i>bool | str</i>) &ndash; whether to load the existing model, or
            which model filename to load
            (<span style="color:#00C000"><b>default</b></span>: true).</li>
            </ul>
            <li><b>max-checkpoints</b> (<i>int > 0</i>) &ndash; maximum number of checkpoints to
            keep (<span style="color:#00C000"><b>default</b></span>: 5).</li>
        summarizer (specification): TensorBoard summarizer configuration with the following
            attributes (<span style="color:#00C000"><b>default</b></span>: no summarizer):
            <ul>
            <li><b>directory</b> (<i>path</i>) &ndash; summarizer directory
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>frequency</b> (<i>int > 0, dict[int > 0]</i>) &ndash; how frequently in
            timesteps to record summaries for act-summaries if specified globally
            (<span style="color:#00C000"><b>default</b></span>: always),
            otherwise specified for act-summaries via "act" in timesteps, for
            observe/experience-summaries via "observe"/"experience" in episodes, and for
            update/variables-summaries via "update"/"variables" in updates
            (<span style="color:#00C000"><b>default</b></span>: never).</li>
            <li><b>flush</b> (<i>int > 0</i>) &ndash; how frequently in seconds to flush the
            summary writer (<span style="color:#00C000"><b>default</b></span>: 10).</li>
            <li><b>max-summaries</b> (<i>int > 0</i>) &ndash; maximum number of summaries to keep
            (<span style="color:#00C000"><b>default</b></span>: 5).</li>
            <li><b>labels</b> (<i>"all" | iter[string]</i>) &ndash; all excluding "*-histogram"
            labels, or list of summaries to record, from the following labels
            (<span style="color:#00C000"><b>default</b></span>: only "graph"):</li>
            <li>"distributions" or "bernoulli", "categorical", "gaussian", "beta":
            distribution-specific parameters</li>
            <li>"dropout": dropout zero fraction</li>
            <li>"entropies" or "entropy", "action-entropies": entropy of policy
            distribution(s)</li>
            <li>"graph": graph summary</li>
            <li>"kl-divergences" or "kl-divergence", "action-kl-divergences": KL-divergence of
            previous and updated polidcy distribution(s)</li>
            <li>"losses" or "loss", "objective-loss", "regularization-loss", "baseline-loss",
            "baseline-objective-loss", "baseline-regularization-loss": loss scalars</li>
            <li>"parameters": parameter scalars</li>
            <li>"relu": ReLU activation zero fraction</li>
            <li>"rewards" or "timestep-reward", "episode-reward", "raw-reward", "empirical-reward",
            "estimated-reward": reward scalar
            </li>
            <li>"update-norm": update norm</li>
            <li>"updates": update mean and variance scalars</li>
            <li>"updates-histogram": update histograms</li>
            <li>"variables": variable mean and variance scalars</li>
            <li>"variables-histogram": variable histograms</li>
            </ul>
        recorder (specification): Experience traces recorder configuration with the following
            attributes (<span style="color:#00C000"><b>default</b></span>: no recorder):
            <ul>
            <li><b>directory</b> (<i>path</i>) &ndash; recorder directory
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>frequency</b> (<i>int > 0</i>) &ndash; how frequently in episodes to record
            traces (<span style="color:#00C000"><b>default</b></span>: every episode).</li>
            <li><b>start</b> (<i>int >= 0</i>) &ndash; how many episodes to skip before starting to
            record traces (<span style="color:#00C000"><b>default</b></span>: 0).</li>
            <li><b>max-traces</b> (<i>int > 0</i>) &ndash; maximum number of traces to keep
            (<span style="color:#00C000"><b>default</b></span>: all).</li>
    """

    def __init__(
        # Environment
        self, states, actions, max_episode_timesteps=None,
        # Network
        network='auto',
        # Optimization
        memory=10000, batch_size=32, update_frequency=4, start_updating=None, learning_rate=3e-4,
        huber_loss=0.0,
        # Reward estimation
        horizon=0, discount=0.99, estimate_terminal=False,  # double_q_model=False !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Target network
        target_sync_frequency=1, target_update_weight=1.0,
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
            agent='dqn',
            states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
            network=network,
            memory=memory, batch_size=batch_size, update_frequency=update_frequency,
            start_updating=start_updating, learning_rate=learning_rate, huber_loss=huber_loss,
            horizon=horizon, discount=discount, estimate_terminal=estimate_terminal,
            target_sync_frequency=target_sync_frequency, target_update_weight=target_update_weight,
            preprocessing=preprocessing,
            exploration=exploration, variable_noise=variable_noise,
            l2_regularization=l2_regularization, entropy_regularization=entropy_regularization,
            name=name, device=device, parallel_interactions=parallel_interactions, seed=seed,
            execution=execution, saver=saver, summarizer=summarizer, recorder=recorder,
            config=config
        )

        # Action value doesn't exist for Beta
        distributions = dict(
            float='gaussian', int=dict(type='categorical', infer_states_value=False)
        )
        policy = dict(network=network, distributions=distributions, temperature=0.0)
        assert max_episode_timesteps is None or \
            memory >= batch_size + max_episode_timesteps + horizon
        memory = dict(type='replay', capacity=memory)
        update = dict(unit='timesteps', batch_size=batch_size, frequency=update_frequency)
        if start_updating is not None:
            update['start'] = start_updating
        optimizer = dict(type='adam', learning_rate=learning_rate)
        objective = dict(type='value', value='action', huber_loss=huber_loss)
        reward_estimation = dict(
            horizon=horizon, discount=discount, estimate_horizon='late',
            estimate_terminal=estimate_terminal, estimate_actions=True
        )
        baseline_policy = policy
        baseline_optimizer = dict(
            type='synchronization', sync_frequency=target_sync_frequency,
            update_weight=target_update_weight
        )
        baseline_objective = None

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
            policy=policy, memory=memory, update=update, optimizer=optimizer, objective=objective,
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective,
            entropy_regularization=entropy_regularization
        )
