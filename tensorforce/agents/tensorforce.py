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
import os
from random import shuffle

import numpy as np

from tensorforce import TensorforceError, util
from tensorforce.agents import Agent
from tensorforce.core import ArrayDict
from tensorforce.core.models import TensorforceModel


class TensorforceAgent(Agent):
    """
    Tensorforce agent (specification key: `tensorforce`).

    Highly configurable agent and basis for a broad class of deep reinforcement learning agents,
    which act according to a policy parametrized by a neural network, leverage a memory module for
    periodic updates based on batches of experience, and optionally employ a baseline/critic/target
    policy for improved reward estimation.

    Args:
        states (specification): States specification
            (<span style="color:#C00000"><b>required</b></span>, better implicitly specified via
            `environment` argument for `Agent.create()`), arbitrarily nested dictionary of state
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
            `environment` argument for `Agent.create()`), arbitrarily nested dictionary of
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
            specified via `environment` argument for `Agent.create()`).

        policy (specification): Policy configuration, see [networks](../modules/networks.html) and
            [policies documentation](../modules/policies.html)
            (<span style="color:#00C000"><b>default</b></span>: action distributions or value
            functions parametrized by an automatically configured network).
        memory (int | specification): Replay memory capacity, or memory configuration, see the
            [memories documentation](../modules/memories.html)
            (<span style="color:#00C000"><b>default</b></span>: minimum capacity recent memory).
        update (int | specification): Model update configuration with the following attributes
            (<span style="color:#C00000"><b>required</b>,
            <span style="color:#00C000"><b>default</b></span>: timesteps batch size</span>):
            <ul>
            <li><b>unit</b> (<i>"timesteps" | "episodes"</i>) &ndash; unit for update attributes
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>batch_size</b>
            (<i><a href="../modules/parameters.html">parameter</a>, int > 0</i>) &ndash;
            size of update batch in number of units
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>frequency</b>
            (<i>"never" | <a href="../modules/parameters.html">parameter</a>, int > 0</i>) &ndash;
            frequency of updates
            (<span style="color:#00C000"><b>default</b></span>: batch_size).</li>
            <li><b>start</b>
            (<i><a href="../modules/parameters.html">parameter</a>, int >= batch_size</i>) &ndash;
            number of units before first update
            (<span style="color:#00C000"><b>default</b></span>: none).</li>
            </ul>
        optimizer (specification): Optimizer configuration, see the
            [optimizers documentation](../modules/optimizers.html)
            (<span style="color:#00C000"><b>default</b></span>: Adam optimizer).
        objective (specification): Optimization objective configuration, see the
            [objectives documentation](../modules/objectives.html)
            (<span style="color:#C00000"><b>required</b></span>).
        reward_estimation (specification): Reward estimation configuration with the following
            attributes (<span style="color:#C00000"><b>required</b></span>):
            <ul>
            <li><b>horizon</b>
            (<i>"episode" | <a href="../modules/parameters.html">parameter</a>, int >= 1</i>)
            &ndash; Horizon of discounted-sum return estimation
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>discount</b>
            (<i><a href="../modules/parameters.html">parameter</a>, 0.0 <= float <= 1.0</i>) &ndash;
            Discount factor of future rewards for discounted-sum return estimation
            (<span style="color:#00C000"><b>default</b></span>: 1.0).</li>
            <li><b>estimate_advantage</b> (<i>bool</i>) &ndash; Whether to use an estimate of the
            advantage (return minus baseline value prediction) instead of the return as learning
            signal
            (<span style="color:#00C000"><b>default</b></span>: false, unless baseline_policy is
            specified but baseline_objective/optimizer are not).</li>
            <li><b>predict_horizon_values</b> (<i>false | "early" | "late"</i>) &ndash; Whether to
            include a baseline prediction of the horizon value as part of the return estimation, and
            if so, whether to compute the horizon value prediction "early" when experiences are
            stored to memory, or "late" when batches of experience are retrieved for the update
            (<span style="color:#00C000"><b>default</b></span>: "late" if baseline_policy or
            baseline_objective are specified, else false).</li>
            <li><b>predict_action_values</b> (<i>bool</i>) &ndash; Whether to predict state-action-
            instead of state-values as horizon values and for advantage estimation
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            <li><b>predict_terminal_values</b> (<i>bool</i>) &ndash; Whether to predict the value of
            terminal states
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            <li><b>return_processing</b> (<i>specification</i>) &ndash; Return processing as layer
            or list of layers, see the [preprocessing documentation](../modules/preprocessing.html)
            (<span style="color:#00C000"><b>default</b></span>: no return processing).</li>
            <li><b>advantage_processing</b> (<i>specification</i>) &ndash; Advantage processing as
            layer or list of layers, see the [preprocessing documentation](../modules/preprocessing.html)
            (<span style="color:#00C000"><b>default</b></span>: no advantage processing).</li>
            </ul>

        baseline (specification): Baseline configuration, policy will be used as baseline if none,
            see [networks](../modules/networks.html) and potentially
            [policies documentation](../modules/policies.html)
            (<span style="color:#00C000"><b>default</b></span>: none).
        baseline_optimizer (specification | <a href="../modules/parameters.html">parameter</a>, float > 0.0):
            Baseline optimizer configuration, see the
            [optimizers documentation](../modules/optimizers.html),
            main optimizer will be used for baseline if none, a float implies none and specifies a
            custom weight for the baseline loss
            (<span style="color:#00C000"><b>default</b></span>: none).
        baseline_objective (specification): Baseline optimization objective configuration, see the
            [objectives documentation](../modules/objectives.html),
            required if baseline optimizer is specified, main objective will be used for baseline if
            baseline objective and optimizer are not specified
            (<span style="color:#00C000"><b>default</b></span>: none).

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

        parallel_interactions (int > 0): Maximum number of parallel interactions to support,
            for instance, to enable multiple parallel episodes, environments or agents within an
            environment
            (<span style="color:#00C000"><b>default</b></span>: 1).
        config (specification): Additional configuration options:
            <ul>
            <li><b>name</b> (<i>string</i>) &ndash; Agent name, used e.g. for TensorFlow scopes and
            saver default filename
            (<span style="color:#00C000"><b>default</b></span>: "agent").
            <li><b>device</b> (<i>string</i>) &ndash; Device name
            (<span style="color:#00C000"><b>default</b></span>: TensorFlow default).
            <li><b>seed</b> (<i>int</i>) &ndash; Random seed to set for Python, NumPy (both set
            globally!) and TensorFlow, environment seed may have to be set separately for fully
            deterministic execution
            (<span style="color:#00C000"><b>default</b></span>: none).</li>
            <li><b>buffer_observe</b> (<i>false | "episode" | int > 0</i>) &ndash; Number of
            timesteps within an episode to buffer before calling the internal observe function, to
            reduce calls to TensorFlow for improved performance
            (<span style="color:#00C000"><b>default</b></span>: configuration-specific maximum
            number which can be buffered without affecting performance).</li>
            <li><b>enable_int_action_masking</b> (<i>bool</i>) &ndash; Whether int action options
            can be masked via an optional "[ACTION-NAME]_mask" state input
            (<span style="color:#00C000"><b>default</b></span>: true).</li>
            <li><b>create_tf_assertions</b> (<i>bool</i>) &ndash; Whether to create internal
            TensorFlow assertion operations
            (<span style="color:#00C000"><b>default</b></span>: true).</li>
            <li><b>eager_mode</b> (<i>bool</i>) &ndash; Whether to run functions eagerly instead of
            running as a traced graph function, can be helpful for debugging
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            <li><b>tf_log_level</b> (<i>int >= 0</i>) &ndash; TensorFlow log level, additional C++
            logging messages can be enabled by setting os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"/"2"
            before importing Tensorforce/TensorFlow
            (<span style="color:#00C000"><b>default</b></span>: 40, only error and critical).</li>
            </ul>
        saver (path | specification): TensorFlow checkpoints directory, or checkpoint manager
            configuration with the following attributes, for periodic implicit saving as alternative
            to explicit saving via agent.save()
            (<span style="color:#00C000"><b>default</b></span>: no saver):
            <ul>
            <li><b>directory</b> (<i>path</i>) &ndash; checkpoint directory
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>filename</b> (<i>string</i>) &ndash; checkpoint filename
            (<span style="color:#00C000"><b>default</b></span>: agent name).</li>
            <li><b>frequency</b> (<i>int > 0</i>) &ndash; how frequently to save a checkpoint
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>unit</b> (<i>"timesteps" | "episodes" | "updates"</i>) &ndash; frequency unit
            (<span style="color:#00C000"><b>default</b></span>: updates).</li>
            <li><b>max_checkpoints</b> (<i>int > 0</i>) &ndash; maximum number of checkpoints to
            keep (<span style="color:#00C000"><b>default</b></span>: 10).</li>
            <li><b>max_hour_frequency</b> (<i>int > 0</i>) &ndash; ignoring max-checkpoints,
            definitely keep a checkpoint in given hour frequency
            (<span style="color:#00C000"><b>default</b></span>: none).</li>
            </ul>
        summarizer (path | specification): TensorBoard summaries directory, or summarizer
            configuration with the following attributes
            (<span style="color:#00C000"><b>default</b></span>: no summarizer):
            <ul>
            <li><b>directory</b> (<i>path</i>) &ndash; summarizer directory
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>filename</b> (<i>path</i>) &ndash; summarizer filename, max_summaries does not
            apply if name specified
            (<span style="color:#00C000"><b>default</b></span>: "summary-%Y%m%d-%H%M%S").</li>
            <li><b>max_summaries</b> (<i>int > 0</i>) &ndash; maximum number of (generically-named)
            summaries to keep
            (<span style="color:#00C000"><b>default</b></span>: 7, number of different colors in
            Tensorboard).</li>
            <li><b>flush</b> (<i>int > 0</i>) &ndash; how frequently in seconds to flush the
            summary writer (<span style="color:#00C000"><b>default</b></span>: 10).</li>
            <li><b>summaries</b> (<i>"all" | iter[string]</i>) &ndash; which summaries to record,
            "all" implies all numerical summaries, so all summaries except "graph"
            (<span style="color:#00C000"><b>default</b></span>: "all"):</li>
            <li>"action-value": value of each action (timestep-based)</li>
            <li>"distribution": distribution parameters like probabilities or mean and stddev
            (timestep-based)</li>
            <li>"entropy": entropy of (per-action) policy distribution(s) (timestep-based)</li>
            <li>"graph": computation graph</li>
            <li>"kl-divergence": KL-divergence of previous and updated (per-action) policy
            distribution(s) (update-based)</li>
            <li>"loss": policy and baseline loss plus loss components (update-based)</li>
            <li>"parameters": parameter values (according to parameter unit)</li>
            <li>"reward": timestep and episode reward, plus intermediate reward/return estimates
            (timestep/episode/update-based)</li>
            <li>"update-norm": global norm of update (update-based)</li>
            <li>"updates": mean and variance of update tensors per variable (update-based)</li>
            <li>"variables": mean of trainable variables tensors (update-based)</li>
            </ul>
        recorder (path | specification): Traces recordings directory, or recorder configuration with
            the following attributes (see
            [record-and-pretrain script](https://github.com/tensorforce/tensorforce/blob/master/examples/record_and_pretrain.py)
            for example application)
            (<span style="color:#00C000"><b>default</b></span>: no recorder):
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
        # Required
        self, states, actions, update, optimizer, objective, reward_estimation,
        # Environment
        max_episode_timesteps=None,
        # Agent
        policy='auto', memory=None,
        # Baseline
        baseline=None, baseline_optimizer=None, baseline_objective=None,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # Preprocessing
        state_preprocessing='linear_normalization', reward_preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Parallel interactions
        parallel_interactions=1,
        # Config, saver, summarizer, recorder
        config=None, saver=None, summarizer=None, recorder=None,
        # Deprecated
        baseline_policy=None, name=None, buffer_observe=None, device=None, seed=None
    ):
        if 'estimate_actions' in reward_estimation:
            raise TensorforceError.deprecated(
                name='Agent', argument='reward_estimation[estimate_actions]',
                replacement='reward_estimation[estimate_action_values]'
            )
        if 'estimate_terminal' in reward_estimation:
            raise TensorforceError.deprecated(
                name='Agent', argument='reward_estimation[estimate_terminal]',
                replacement='reward_estimation[estimate_terminals]'
            )
        if baseline_policy is not None:
            raise TensorforceError.deprecated(
                name='Agent', argument='baseline_policy', replacement='baseline'
            )
        if name is not None:
            raise TensorforceError.deprecated(
                name='Agent', argument='name', replacement='config[name]'
            )
        if buffer_observe is not None:
            raise TensorforceError.deprecated(
                name='Agent', argument='buffer_observe', replacement='config[buffer_observe]'
            )
        if device is not None:
            raise TensorforceError.deprecated(
                name='Agent', argument='device', replacement='config[device]'
            )
        if seed is not None:
            raise TensorforceError.deprecated(
                name='Agent', argument='seed', replacement='config[seed]'
            )

        if not hasattr(self, 'spec'):
            self.spec = OrderedDict(
                agent='tensorforce',
                # Environment
                states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
                # Agent
                policy=policy, memory=memory, update=update, optimizer=optimizer,
                objective=objective, reward_estimation=reward_estimation,
                # Baseline
                baseline=baseline, baseline_optimizer=baseline_optimizer,
                baseline_objective=baseline_objective,
                # Regularization
                l2_regularization=l2_regularization, entropy_regularization=entropy_regularization,
                # Preprocessing
                state_preprocessing=state_preprocessing, reward_preprocessing=reward_preprocessing,
                # Exploration
                exploration=exploration, variable_noise=variable_noise,
                # Parallel interactions
                parallel_interactions=parallel_interactions,
                # Config, saver, summarizer, recorder
                config=config, saver=saver, summarizer=summarizer, recorder=recorder
            )

        if memory is None:
            memory = dict(type='recent')

        if isinstance(update, int):
            update = dict(unit='timesteps', batch_size=update)

        if config is None:
            config = dict()
        else:
            config = dict(config)

        # TODO: should this change if summarizer is specified?
        if parallel_interactions > 1:
            if 'buffer_observe' not in config:
                if max_episode_timesteps is None:
                    raise TensorforceError.required(
                        name='Agent', argument='max_episode_timesteps',
                        condition='parallel_interactions > 1'
                    )
                config['buffer_observe'] = 'episode'
            # elif config['buffer_observe'] < max_episode_timesteps:
            #     raise TensorforceError.value(
            #         name='Agent', argument='config[buffer_observe]',
            #         hint='< max_episode_timesteps', condition='parallel_interactions > 1'
            #     )

        elif update['unit'] == 'timesteps':
            update_frequency = update.get('frequency', update['batch_size'])
            if 'buffer_observe' not in config:
                if isinstance(update_frequency, int):
                    config['buffer_observe'] = update_frequency
                else:
                    config['buffer_observe'] = 1
            elif isinstance(update_frequency, int) and (
                config['buffer_observe'] == 'episode' or config['buffer_observe'] > update_frequency
            ):
                raise TensorforceError.value(
                    name='Agent', argument='config[buffer_observe]', value=config['buffer_observe'],
                    hint='> update[frequency]', condition='update[unit] = "timesteps"'
                )

        elif update['unit'] == 'episodes':
            if 'buffer_observe' not in config:
                config['buffer_observe'] = 'episode'

        # reward_estimation = dict(reward_estimation)
        # if reward_estimation['horizon'] == 'episode':
        #     if max_episode_timesteps is None:
        #         raise TensorforceError.required(
        #             name='Agent', argument='max_episode_timesteps',
        #             condition='reward_estimation[horizon] = "episode"'
        #         )
        #     reward_estimation['horizon'] = max_episode_timesteps

        super().__init__(
            states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
            parallel_interactions=parallel_interactions, config=config, recorder=recorder
        )

        self.model = TensorforceModel(
            states=self.states_spec, actions=self.actions_spec,
            max_episode_timesteps=self.max_episode_timesteps,
            policy=policy, memory=memory, update=update, optimizer=optimizer, objective=objective,
            reward_estimation=reward_estimation,
            baseline=baseline, baseline_optimizer=baseline_optimizer,
            baseline_objective=baseline_objective,
            l2_regularization=l2_regularization, entropy_regularization=entropy_regularization,
            state_preprocessing=state_preprocessing, reward_preprocessing=reward_preprocessing,
            exploration=exploration, variable_noise=variable_noise,
            parallel_interactions=self.parallel_interactions,
            config=self.config, saver=saver, summarizer=summarizer
        )

    def experience(self, states, actions, terminal, reward, internals=None):
        """
        Feed experience traces.

        See the [act-experience-update script](https://github.com/tensorforce/tensorforce/blob/master/examples/act_experience_update_interface.py)
        for an example application as part of the act-experience-update interface, which is an
        alternative to the act-observe interaction pattern.

        Args:
            states (dict[array[state]]): Dictionary containing arrays of states
                (<span style="color:#C00000"><b>required</b></span>).
            actions (dict[array[action]]): Dictionary containing arrays of actions
                (<span style="color:#C00000"><b>required</b></span>).
            terminal (array[bool]): Array of terminals
                (<span style="color:#C00000"><b>required</b></span>).
            reward (array[float]): Array of rewards
                (<span style="color:#C00000"><b>required</b></span>).
            internals (dict[state]): Dictionary containing arrays of internal agent states
                (<span style="color:#C00000"><b>required</b></span> if agent has internal states).
        """
        if not all(len(buffer) == 0 for buffer in self.terminal_buffer):
            raise TensorforceError(message="Calling agent.experience is not possible mid-episode.")

        # Process states input and infer batching structure
        states, batched, num_instances, is_iter_of_dicts, input_type = self._process_states_input(
            states=states, function_name='Agent.experience'
        )

        if is_iter_of_dicts:
            # Input structure iter[dict[input]]

            # Internals
            if internals is None:
                internals = ArrayDict(self.initial_internals())
                internals = internals.fmap(function=(lambda x: np.repeat(np.expand_dims(x, axis=0), repeats=num_instances, axis=0)))
            elif not isinstance(internals, (tuple, list)):
                raise TensorforceError.type(
                    name='Agent.experience', argument='internals', dtype=type(internals),
                    hint='is not tuple/list'
                )
            else:
                internals = [ArrayDict(internal) for internal in internals]
                internals = internals[0].fmap(
                    function=(lambda *xs: np.stack(xs, axis=0)), zip_values=internals[1:]
                )

            # Actions
            if isinstance(actions, np.ndarray):
                actions = ArrayDict(singleton=actions)
            elif not isinstance(actions, (tuple, list)):
                raise TensorforceError.type(
                    name='Agent.experience', argument='actions', dtype=type(actions),
                    hint='is not tuple/list'
                )
            elif not isinstance(actions[0], dict):
                actions = ArrayDict(singleton=np.asarray(actions))
            else:
                actions = [ArrayDict(action) for action in actions]
                actions = actions[0].fmap(
                    function=(lambda *xs: np.stack(xs, axis=0)), zip_values=actions[1:]
                )

        else:
            # Input structure dict[iter[input]]

            # Internals
            if internals is None:
                internals = ArrayDict(self.initial_internals())
                internals = internals.fmap(function=(lambda x: np.tile(np.expand_dims(x, axis=0), reps=(num_instances,))))
            elif not isinstance(internals, dict):
                raise TensorforceError.type(
                    name='Agent.experience', argument='internals', dtype=type(internals),
                    hint='is not dict'
                )
            else:
                internals = ArrayDict(internals)

            # Actions
            if not isinstance(actions, np.ndarray):
                actions = ArrayDict(singleton=actions)
            elif not isinstance(actions, dict):
                raise TensorforceError.type(
                    name='Agent.experience', argument='actions', dtype=type(actions),
                    hint='is not dict'
                )
            else:
                actions = ArrayDict(actions)

        # Expand inputs if not batched
        if not batched:
            internals = internals.fmap(function=(lambda x: np.expand_dims(x, axis=0)))
            actions = actions.fmap(function=(lambda x: np.expand_dims(x, axis=0)))
            terminal = np.asarray([terminal])
            reward = np.asarray([reward])
        else:
            terminal = np.asarray(terminal)
            reward = np.asarray(reward)

        # Check number of inputs
        for name, internal in internals.items():
            if internal.shape[0] != num_instances:
                raise TensorforceError.value(
                    name='Agent.experience', argument='len(internals[{}])'.format(name),
                    value=internal.shape[0], hint='!= len(states)'
                )
        for name, action in actions.items():
            if action.shape[0] != num_instances:
                raise TensorforceError.value(
                    name='Agent.experience', argument='len(actions[{}])'.format(name),
                    value=action.shape[0], hint='!= len(states)'
                )
        if terminal.shape[0] != num_instances:
            raise TensorforceError.value(
                name='Agent.experience', argument='len(terminal)'.format(name),
                value=terminal.shape[0], hint='!= len(states)'
            )
        if reward.shape[0] != num_instances:
            raise TensorforceError.value(
                name='Agent.experience', argument='len(reward)'.format(name),
                value=reward.shape[0], hint='!= len(states)'
            )

        def function(name, spec):
            auxiliary = ArrayDict()
            if self.config.enable_int_action_masking and spec.type == 'int' and \
                    spec.num_values is not None:
                if name is None:
                    name = 'action'
                # Mask, either part of states or default all true
                auxiliary['mask'] = states.pop(name + '_mask', np.ones(
                    shape=(num_instances,) + spec.shape + (spec.num_values,), dtype=spec.np_type()
                ))
            return auxiliary

        auxiliaries = self.actions_spec.fmap(function=function, cls=ArrayDict, with_names=True)
        if self.states_spec.is_singleton() and not states.is_singleton():
            states[None] = states.pop('state')

        # Convert terminal to int if necessary
        if terminal.dtype is util.np_dtype(dtype='bool'):
            zeros = np.zeros_like(terminal, dtype=util.np_dtype(dtype='int'))
            ones = np.ones_like(terminal, dtype=util.np_dtype(dtype='int'))
            terminal = np.where(terminal, ones, zeros)

        if terminal[-1] == 0:
            raise TensorforceError(message="Agent.experience() requires full episodes as input.")

        # Batch experiences split into episodes and at most size buffer_observe
        last = 0
        for index in range(1, len(terminal) + 1):
            if terminal[index - 1] == 0:
                continue

            function = (lambda x: x[last: index])
            states_batch = states.fmap(function=function)
            internals_batch = internals.fmap(function=function)
            auxiliaries_batch = auxiliaries.fmap(function=function)
            actions_batch = actions.fmap(function=function)
            terminal_batch = function(terminal)
            reward_batch = function(reward)
            last = index

            # Inputs to tensors
            states_batch = self.states_spec.to_tensor(value=states_batch, batched=True)
            internals_batch = self.internals_spec.to_tensor(
                value=internals_batch, batched=True, recover_empty=True
            )
            auxiliaries_batch = self.auxiliaries_spec.to_tensor(
                value=auxiliaries_batch, batched=True
            )
            actions_batch = self.actions_spec.to_tensor(value=actions_batch, batched=True)
            terminal_batch = self.terminal_spec.to_tensor(value=terminal_batch, batched=True)
            reward_batch = self.reward_spec.to_tensor(value=reward_batch, batched=True)

            # Model.experience()
            timesteps, episodes = self.model.experience(
                states=states_batch, internals=internals_batch, auxiliaries=auxiliaries_batch,
                actions=actions_batch, terminal=terminal_batch, reward=reward_batch
            )
            self.timesteps = timesteps.numpy().item()
            self.episodes = episodes.numpy().item()

        if self.model.saver is not None:
            self.model.save()

    def update(self, query=None, **kwargs):
        """
        Perform an update.

        See the [act-experience-update script](https://github.com/tensorforce/tensorforce/blob/master/examples/act_experience_update_interface.py)
        for an example application as part of the act-experience-update interface, which is an
        alternative to the act-observe interaction pattern.
        """
        updates = self.model.update()
        self.updates = updates.numpy().item()

        if self.model.saver is not None:
            self.model.save()

    def pretrain(self, directory, num_iterations, num_traces=1, num_updates=1, extension='.npz'):
        """
        Simple pretraining approach as a combination of `experience()` and `update`, akin to
        behavioral cloning, using experience traces obtained e.g. via recording agent interactions
        ([see documentation](https://tensorforce.readthedocs.io/en/latest/basics/features.html#record-pretrain)).

        For the given number of iterations, load the given number of trace files (which each contain
        recorder[frequency] episodes), feed the experience to the agent's internal memory, and
        subsequently trigger the given number of updates (which will use the experience in the
        internal memory, fed in this or potentially previous iterations).

        See the [record-and-pretrain script](https://github.com/tensorforce/tensorforce/blob/master/examples/record_and_pretrain.py)
        for an example application.

        Args:
            directory (path): Directory with experience traces, e.g. obtained via recorder; episode
                length has to be consistent with agent configuration
                (<span style="color:#C00000"><b>required</b></span>).
            num_iterations (int > 0): Number of iterations consisting of loading new traces and
                performing multiple updates
                (<span style="color:#C00000"><b>required</b></span>).
            num_traces (int > 0): Number of traces to load per iteration; has to at least satisfy
                the update batch size
                (<span style="color:#00C000"><b>default</b></span>: 1).
            num_updates (int > 0): Number of updates per iteration
                (<span style="color:#00C000"><b>default</b></span>: 1).
            extension (str): Traces file extension to filter the given directory for
                (<span style="color:#00C000"><b>default</b></span>: ".npz").
        """
        if not os.path.isdir(directory):
            raise TensorforceError.value(
                name='agent.pretrain', argument='directory', value=directory
            )
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1] == extension
        )
        indices = list(range(len(files)))

        for _ in range(num_iterations):
            shuffle(indices)
            if num_traces is None:
                selection = indices
            else:
                selection = indices[:num_traces]

            batch = None
            for index in selection:
                trace = ArrayDict(np.load(files[index]))
                if batch is None:
                    batch = trace
                else:
                    batch = batch.fmap(
                        function=(lambda x, y: np.concatenate([x, y], axis=0)), zip_values=(trace,)
                    )

            for name, value in batch.pop('auxiliaries', dict()).items():
                assert name.endswith('/mask')
                batch['states'][name[:-5] + '_mask'] = value

            self.experience(**batch.to_kwargs())
            for _ in range(num_updates):
                self.update()
            # TODO: self.obliviate()
