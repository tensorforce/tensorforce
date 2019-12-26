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
import os
from random import shuffle

import numpy as np

from tensorforce import TensorforceError, util
from tensorforce.agents import Agent
from tensorforce.core.models import TensorforceModel


class TensorforceAgent(Agent):
    """
    Tensorforce agent (specification key: `tensorforce`).

    Base class for a broad class of deep reinforcement learning agents, which act according to a
    policy parametrized by a neural network, leverage a memory module for periodic updates based on
    batches of experience, and optionally employ a baseline/critic/target policy for improved
    reward estimation.

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

        policy (specification): Policy configuration, see [policies](../modules/policies.html)
            (<span style="color:#00C000"><b>default</b></span>: "default", action distributions
            parametrized by an automatically configured network).
        memory (int | specification): Memory configuration, see
            [memories](../modules/memories.html)
            (<span style="color:#00C000"><b>default</b></span>: replay memory with given or
            inferred capacity).
        update (int | specification): Model update configuration with the following attributes
            (<span style="color:#C00000"><b>required</b>,
            <span style="color:#00C000"><b>default</b></span>: timesteps batch size</span>):
            <ul>
            <li><b>unit</b> (<i>"timesteps" | "episodes"</i>) &ndash; unit for update attributes
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>batch_size</b> (<i>parameter, long > 0</i>) &ndash; size of update batch in
            number of units (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>frequency</b> (<i>"never" | parameter, long > 0</i>) &ndash; frequency of
            updates (<span style="color:#00C000"><b>default</b></span>: batch_size).</li>
            <li><b>start</b> (<i>parameter, long >= batch_size</i>) &ndash; number of units
            before first update (<span style="color:#00C000"><b>default</b></span>: 0).</li>
            </ul>
        optimizer (specification): Optimizer configuration, see
            [optimizers](../modules/optimizers.html)
            (<span style="color:#00C000"><b>default</b></span>: Adam optimizer).
        objective (specification): Optimization objective configuration, see
            [objectives](../modules/objectives.html)
            (<span style="color:#C00000"><b>required</b></span>).
        reward_estimation (specification): Reward estimation configuration with the following
            attributes (<span style="color:#C00000"><b>required</b></span>):
            <ul>
            <li><b>horizon</b> (<i>"episode" | parameter, long >= 0</i>) &ndash; Horizon of
            discounted-sum reward estimation
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>discount</b> (<i>parameter, 0.0 <= float <= 1.0</i>) &ndash; Discount factor for
            future rewards of discounted-sum reward estimation
            (<span style="color:#00C000"><b>default</b></span>: 1.0).</li>
            <li><b>estimate_horizon</b> (<i>false | "early" | "late"</i>) &ndash; Whether to
            estimate the value of horizon states, and if so, whether to estimate early when
            experience is stored, or late when it is retrieved
            (<span style="color:#00C000"><b>default</b></span>: "late" if any of the baseline_*
            arguments is specified, else false).</li>
            <li><b>estimate_actions</b> (<i>bool</i>) &ndash; Whether to estimate state-action
            values instead of state values
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            <li><b>estimate_terminal</b> (<i>bool</i>) &ndash; Whether to estimate the value of
            (real) terminal states (<span style="color:#00C000"><b>default</b></span>: false).</li>
            <li><b>estimate_advantage</b> (<i>bool</i>) &ndash; Whether to estimate the advantage
            by subtracting the current estimate
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            </ul>

        baseline_policy (specification): Baseline policy configuration, main policy will be used as
            baseline if none
            (<span style="color:#00C000"><b>default</b></span>: none).
        baseline_optimizer (float > 0.0 | specification): Baseline optimizer configuration, see
            [optimizers](../modules/optimizers.html), main optimizer will be used for baseline if
            none, a float implies none and specifies a custom weight for the baseline loss
            (<span style="color:#00C000"><b>default</b></span>: none).
        baseline_objective (specification): Baseline optimization objective configuration, see
            [objectives](../modules/objectives.html), main objective will be used for baseline if
            none (<span style="color:#00C000"><b>default</b></span>: none).

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
        buffer_observe (bool | int > 0): Maximum number of timesteps within an episode to buffer
            before executing internal observe operations, to reduce calls to TensorFlow for
            improved performance
            (<span style="color:#00C000"><b>default</b></span>: max_episode_timesteps or 1000,
            unless summarizer specified).
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
            <li><b>start</b> (<i>int >= 0</i>) &ndash; how many episodes to skip before starting to
            record traces (<span style="color:#00C000"><b>default</b></span>: 0).</li>
            <li><b>max-traces</b> (<i>int > 0</i>) &ndash; maximum number of traces to keep
            (<span style="color:#00C000"><b>default</b></span>: all).</li>
    """

    def __init__(
        self,
        # --- required ---
        # Environment
        states, actions,
        # Agent
        update, objective, reward_estimation,
        # --- default ---
        # Environment
        max_episode_timesteps=None,
        # Agent
        policy='default', memory=None, optimizer='adam',
        # Baseline
        baseline_policy=None, baseline_optimizer=None, baseline_objective=None,
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, buffer_observe=True, seed=None,
        execution=None, saver=None, summarizer=None, recorder=None, config=None
    ):
        if not hasattr(self, 'spec'):
            self.spec = OrderedDict(
                agent='tensorforce',
                states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
                policy=policy, memory=memory, update=update, optimizer=optimizer,
                objective=objective, reward_estimation=reward_estimation,
                baseline_policy=baseline_policy, baseline_optimizer=baseline_optimizer,
                baseline_objective=baseline_objective,
                preprocessing=preprocessing,
                exploration=exploration, variable_noise=variable_noise,
                l2_regularization=l2_regularization, entropy_regularization=entropy_regularization,
                name=name, device=device, parallel_interactions=parallel_interactions,
                buffer_observe=buffer_observe, seed=seed, execution=execution, saver=saver,
                summarizer=summarizer, recorder=recorder, config=config
            )

        if buffer_observe is True and parallel_interactions == 1 and summarizer is not None:
            buffer_observe = False

        if isinstance(update, int) or update['unit'] == 'timesteps':
            if buffer_observe is not True or parallel_interactions > 1:
                TensorforceError.unexpected()
            buffer_observe = False

        super().__init__(
            states=states, actions=actions, max_episode_timesteps=max_episode_timesteps,
            parallel_interactions=parallel_interactions, buffer_observe=buffer_observe, seed=seed,
            recorder=recorder
        )

        if isinstance(update, int):
            update = dict(unit='timesteps', batch_size=update)

        reward_estimation = dict(reward_estimation)
        if reward_estimation['horizon'] == 'episode':
            if max_episode_timesteps is None:
                raise TensorforceError.unexpected()
            reward_estimation['horizon'] = max_episode_timesteps
        if 'capacity' not in reward_estimation:
            # TODO: Doesn't take network horizon into account, needs to be set internally to max
            # if isinstance(reward_estimation['horizon'], int):
            #     reward_estimation['capacity'] = max(
            #         self.buffer_observe, reward_estimation['horizon'] + 2
            #     )
            if max_episode_timesteps is None:
                raise TensorforceError.unexpected()
            if isinstance(reward_estimation['horizon'], int):
                reward_estimation['capacity'] = max(
                    max_episode_timesteps, reward_estimation['horizon']
                )
            else:
                reward_estimation['capacity'] = max_episode_timesteps
        self.experience_size = reward_estimation['capacity']

        if memory is None:
            # predecessor/successor?
            if max_episode_timesteps is None or not isinstance(update['batch_size'], int) \
                    or not isinstance(reward_estimation['horizon'], int):
                raise TensorforceError.unexpected()
            if update['unit'] == 'timesteps':
                memory = update['batch_size'] + max_episode_timesteps + \
                    reward_estimation['horizon']
                # memory = ceil(update['batch_size'] / max_episode_timesteps) * max_episode_timesteps
                # memory += int(update['batch_size'] / max_episode_timesteps >= 1.0)
            elif update['unit'] == 'episodes':
                memory = update['batch_size'] * max_episode_timesteps + \
                    max(reward_estimation['horizon'], max_episode_timesteps)
            memory = max(memory, min(self.buffer_observe, max_episode_timesteps))

        self.model = TensorforceModel(
            # Model
            name=name, device=device, parallel_interactions=self.parallel_interactions,
            buffer_observe=self.buffer_observe, seed=seed, execution=execution, saver=saver,
            summarizer=summarizer, config=config, states=self.states_spec,
            actions=self.actions_spec, preprocessing=preprocessing, exploration=exploration,
            variable_noise=variable_noise, l2_regularization=l2_regularization,
            # TensorforceModel
            policy=policy, memory=memory, update=update, optimizer=optimizer, objective=objective,
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective,
            entropy_regularization=entropy_regularization
        )

        assert max_episode_timesteps is None or self.model.memory.capacity > max_episode_timesteps

    def experience(self, states, actions, terminal, reward, internals=None, query=None, **kwargs):
        """
        Feed experience traces.

        Args:
            states (dict[array[state]): Dictionary containing arrays of states
                (<span style="color:#C00000"><b>required</b></span>).
            actions (dict[array[action]]): Dictionary containing arrays of actions
                (<span style="color:#C00000"><b>required</b></span>).
            terminal (array[bool]): Array of terminals
                (<span style="color:#C00000"><b>required</b></span>).
            reward (array[float]): Array of rewards
                (<span style="color:#C00000"><b>required</b></span>).
            internals (dict[state]): Dictionary containing arrays of internal states
                (<span style="color:#00C000"><b>default</b></span>: no internal states).
            query (list[str]): Names of tensors to retrieve
                (<span style="color:#00C000"><b>default</b></span>: none).
            kwargs: Additional input values, for instance, for dynamic hyperparameters.
        """
        assert (self.buffer_indices == 0).all()
        assert util.reduce_all(predicate=util.not_nan_inf, xs=states)
        assert internals is None  # or util.reduce_all(predicate=util.not_nan_inf, xs=internals)
        assert util.reduce_all(predicate=util.not_nan_inf, xs=actions)
        assert util.reduce_all(predicate=util.not_nan_inf, xs=reward)

        # Auxiliaries
        auxiliaries = OrderedDict()
        if isinstance(states, dict):
            for name, spec in self.actions_spec.items():
                if spec['type'] == 'int' and name + '_mask' in states:
                    auxiliaries[name + '_mask'] = np.asarray(states.pop(name + '_mask'))
        auxiliaries = util.fmap(function=np.asarray, xs=auxiliaries, depth=1)

        # Normalize states/actions dictionaries
        states = util.normalize_values(
            value_type='state', values=states, values_spec=self.states_spec
        )
        if internals is None:
            internals = OrderedDict()
        actions = util.normalize_values(
            value_type='action', values=actions, values_spec=self.actions_spec
        )

        if isinstance(terminal, (bool, int)):
            states = util.fmap(function=(lambda x: [x]), xs=states, depth=1)
            actions = util.fmap(function=(lambda x: [x]), xs=actions, depth=1)
            terminal = [terminal]
            reward = [reward]

        states = util.fmap(function=np.asarray, xs=states, depth=1)
        actions = util.fmap(function=np.asarray, xs=actions, depth=1)

        if isinstance(terminal, np.ndarray):
            if terminal.dtype is util.np_dtype(dtype='bool'):
                zeros = np.zeros_like(terminal, dtype=util.np_dtype(dtype='long'))
                ones = np.ones_like(terminal, dtype=util.np_dtype(dtype='long'))
                terminal = np.where(terminal, ones, zeros)
        else:
            terminal = np.asarray([int(x) if isinstance(x, bool) else x for x in terminal])
        reward = np.asarray(reward)

        # Batch experiences split into episodes and at most size buffer_observe
        last = 0
        for index in range(1, len(terminal) + 1):
            if terminal[index - 1] == 0 and index - last < self.experience_size:
                continue

            # Include terminal in batch if possible
            if index < len(terminal) and terminal[index - 1] == 0 and terminal[index] > 0 and \
                    index - last < self.experience_size:
                index += 1

            function = (lambda x: x[last: index])
            states_batch = util.fmap(function=function, xs=states, depth=1)
            internals_batch = util.fmap(function=function, xs=internals, depth=1)
            auxiliaries_batch = util.fmap(function=function, xs=auxiliaries, depth=1)
            actions_batch = util.fmap(function=function, xs=actions, depth=1)
            terminal_batch = terminal[last: index]
            reward_batch = reward[last: index]
            last = index

            # Model.experience()
            if query is None:
                self.timesteps, self.episodes, self.updates = self.model.experience(
                    states=states_batch, internals=internals_batch,
                    auxiliaries=auxiliaries_batch, actions=actions_batch, terminal=terminal_batch,
                    reward=reward_batch, **kwargs
                )

            else:
                self.timesteps, self.episodes, self.updates, queried = self.model.experience(
                    states=states_batch, internals=internals_batch,
                    auxiliaries=auxiliaries_batch, actions=actions_batch, terminal=terminal_batch,
                    reward=reward_batch, query=query, **kwargs
                )

        if query is not None:
            return queried

    def update(self, query=None, **kwargs):
        """
        Perform an update.

        Args:
            query (list[str]): Names of tensors to retrieve
                (<span style="color:#00C000"><b>default</b></span>: none).
            kwargs: Additional input values, for instance, for dynamic hyperparameters.
        """
        # Model.update()
        if query is None:
            self.timesteps, self.episodes, self.updates = self.model.update(**kwargs)

        else:
            self.timesteps, self.episodes, self.updates, queried = self.model.update(
                query=query, **kwargs
            )
            return queried

    def pretrain(self, directory, num_iterations, num_traces=1, num_updates=1):
        """
        Pretrain from experience traces.

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
        """
        if not os.path.isdir(directory):
            raise TensorforceError.unexpected()
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.startswith('trace-')
        )
        indices = list(range(len(files)))

        for _ in range(num_iterations):
            shuffle(indices)
            if num_traces is None:
                selection = indices
            else:
                selection = indices[:num_traces]

            states = OrderedDict(((name, list()) for name in self.states_spec))
            for name, spec in self.actions_spec.items():
                if spec['type'] == 'int':
                    states[name + '_mask'] = list()
            actions = OrderedDict(((name, list()) for name in self.actions_spec))
            terminal = list()
            reward = list()
            for index in selection:
                trace = np.load(files[index])
                for name in states:
                    states[name].append(trace[name])
                for name in actions:
                    actions[name].append(trace[name])
                terminal.append(trace['terminal'])
                reward.append(trace['reward'])

            states = util.fmap(function=np.concatenate, xs=states, depth=1)
            actions = util.fmap(function=np.concatenate, xs=actions, depth=1)
            terminal = np.concatenate(terminal)
            reward = np.concatenate(reward)

            self.experience(states=states, actions=actions, terminal=terminal, reward=reward)
            for _ in range(num_updates):
                self.update()
            # TODO: self.obliviate()
