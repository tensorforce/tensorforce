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

import time
from tqdm import tqdm

import numpy as np

from tensorforce import TensorforceError, util
from tensorforce.agents import Agent
from tensorforce.environments import Environment


class ParallelRunner(object):
    """
    Tensorforce parallel runner utility.

    Args:
        agent (specification | Agent object): Agent specification or object, the latter is not
            closed automatically as part of `runner.close()`
            (<span style="color:#C00000"><b>required</b></span>).
        environment (specification | Environment object): Environment specification or object, the
            latter is not closed automatically as part of `runner.close()`
            (<span style="color:#C00000"><b>required</b></span>, or alternatively `environments`).
        num_parallel (int > 0): Number of parallel environment instances to run
            (<span style="color:#C00000"><b>required</b></span>, or alternatively `environments`).
        environments (list[specification | Environment object]): Environment specifications or
            objects, the latter are not closed automatically as part of `runner.close()`
            (<span style="color:#C00000"><b>required</b></span>, or alternatively `environment` and
            `num_parallel`).
        max_episode_timesteps (int > 0): Maximum number of timesteps per episode, overwrites the
            environment default if defined
            (<span style="color:#00C000"><b>default</b></span>: environment default).
        evaluation_environment (specification | Environment object): Evaluation environment or
            object, the latter is not closed automatically as part of `runner.close()`
            (<span style="color:#00C000"><b>default</b></span>: none).
        save_best_agent (string): Directory to save the best version of the agent according to the
            evaluation
            (<span style="color:#00C000"><b>default</b></span>: best agent is not saved).
    """

    def __init__(
        self, agent, environment=None, num_parallel=None, environments=None,
        max_episode_timesteps=None, evaluation_environment=None, save_best_agent=None
    ):
        self.environments = list()
        if environment is None:
            assert num_parallel is None and environments is not None
            if not util.is_iterable(x=environments):
                raise TensorforceError.type(
                    name='parallel-runner', argument='environments', value=environments
                )
            elif len(environments) == 0:
                raise TensorforceError.value(
                    name='parallel-runner', argument='environments', value=environments
                )
            num_parallel = len(environments)
            environment = environments[0]
            self.is_environment_external = isinstance(environment, Environment)
            environment = Environment.create(
                environment=environment, max_episode_timesteps=max_episode_timesteps
            )
            states = environment.states()
            actions = environment.actions()
            self.environments.append(environment)
            for environment in environments[1:]:
                assert isinstance(environment, Environment) == self.is_environment_external
                environment = Environment.create(
                    environment=environment, max_episode_timesteps=max_episode_timesteps
                )
                assert environment.states() == states
                assert environment.actions() == actions
                self.environments.append(environment)

        else:
            assert num_parallel is not None and environments is None
            assert not isinstance(environment, Environment)
            self.is_environment_external = False
            for _ in range(num_parallel):
                environment = Environment.create(
                    environment=environment, max_episode_timesteps=max_episode_timesteps
                )
                self.environments(environment)

        if evaluation_environment is None:
            self.evaluation_environment = None
        else:
            self.is_eval_environment_external = isinstance(evaluation_environment, Environment)
            self.evaluation_environment = Environment.create(
                environment=evaluation_environment, max_episode_timesteps=max_episode_timesteps
            )
            assert self.evaluation_environment.states() == environment.states()
            assert self.evaluation_environment.actions() == environment.actions()

        self.is_agent_external = isinstance(agent, Agent)
        kwargs = dict(parallel_interactions=num_parallel)
        self.agent = Agent.create(agent=agent, environment=environment, **kwargs)
        self.save_best_agent = save_best_agent

        self.episode_rewards = list()
        self.episode_timesteps = list()
        self.episode_seconds = list()
        self.episode_agent_seconds = list()
        self.evaluation_rewards = list()
        self.evaluation_timesteps = list()
        self.evaluation_seconds = list()
        self.evaluation_agent_seconds = list()

    def close(self):
        if hasattr(self, 'tqdm'):
            self.tqdm.close()
        if not self.is_agent_external:
            self.agent.close()
        if not self.is_environment_external:
            for environment in self.environments:
                environment.close()
        if self.evaluation_environment is not None and not self.is_eval_environment_external:
            self.evaluation_environment.close()
        self.agent.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(
        self,
        # General
        num_episodes=None, num_timesteps=None, num_updates=None, num_sleep_secs=0.01,
        sync_timesteps=False, sync_episodes=False,
        # Callback
        callback=None, callback_episode_frequency=None, callback_timestep_frequency=None,
        # Tqdm
        use_tqdm=True, mean_horizon=1,
        # Evaluation
        evaluation_callback=None
    ):
        # General
        if num_episodes is None:
            self.num_episodes = float('inf')
        else:
            self.num_episodes = num_episodes
        if num_timesteps is None:
            self.num_timesteps = float('inf')
        else:
            self.num_timesteps = num_timesteps
        if num_updates is None:
            self.num_updates = float('inf')
        else:
            self.num_updates = num_updates
        self.num_sleep_secs = num_sleep_secs
        self.sync_timesteps = sync_timesteps
        self.sync_episodes = sync_episodes

        # Callback
        assert callback_episode_frequency is None or callback_timestep_frequency is None
        if callback_episode_frequency is None and callback_timestep_frequency is None:
            callback_episode_frequency = 1
        if callback_episode_frequency is None:
            self.callback_episode_frequency = float('inf')
        else:
            self.callback_episode_frequency = callback_episode_frequency
        if callback_timestep_frequency is None:
            self.callback_timestep_frequency = float('inf')
        else:
            self.callback_timestep_frequency = callback_timestep_frequency
        if callback is None:
            self.callback = (lambda r, p: True)
        elif util.is_iterable(x=callback):
            def sequential_callback(runner, parallel):
                result = True
                for fn in callback:
                    x = fn(runner, parallel)
                    if isinstance(result, bool):
                        result = result and x
                return result
            self.callback = sequential_callback
        else:
            def boolean_callback(runner, parallel):
                result = callback(runner, parallel)
                if isinstance(result, bool):
                    return result
                else:
                    return True
            self.callback = boolean_callback

        # Timestep/episode/update counter
        self.timesteps = 0
        self.episodes = 0
        self.updates = 0

        # Tqdm
        if use_tqdm:
            if hasattr(self, 'tqdm'):
                self.tqdm.close()

            assert self.num_episodes != float('inf') or self.num_timesteps != float('inf')
            inner_callback = self.callback

            if self.num_episodes != float('inf'):
                # Episode-based tqdm (default option if both num_episodes and num_timesteps set)
                assert self.num_episodes != float('inf')
                bar_format = (
                    '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, reward={postfix[0]:.2f}, ts/ep='
                    '{postfix[1]}, sec/ep={postfix[2]:.2f}, ms/ts={postfix[3]:.1f}, agent='
                    '{postfix[4]:.1f}%]'
                )
                postfix = [0.0, 0, 0.0, 0.0, 0.0]
                self.tqdm = tqdm(
                    desc='Episodes', total=self.num_episodes, bar_format=bar_format,
                    initial=self.episodes, postfix=postfix
                )
                self.tqdm_last_update = self.episodes

                def tqdm_callback(runner, parallel):
                    mean_reward = float(np.mean(runner.episode_rewards[-mean_horizon:]))
                    mean_ts_per_ep = int(np.mean(runner.episode_timesteps[-mean_horizon:]))
                    mean_sec_per_ep = float(np.mean(runner.episode_seconds[-mean_horizon:]))
                    mean_agent_sec = float(np.mean(runner.episode_agent_seconds[-mean_horizon:]))
                    mean_ms_per_ts = mean_sec_per_ep * 1000.0 / mean_ts_per_ep
                    mean_rel_agent = mean_agent_sec * 100.0 / mean_sec_per_ep
                    runner.tqdm.postfix[0] = mean_reward
                    runner.tqdm.postfix[1] = mean_ts_per_ep
                    runner.tqdm.postfix[2] = mean_sec_per_ep
                    runner.tqdm.postfix[3] = mean_ms_per_ts
                    runner.tqdm.postfix[4] = mean_rel_agent
                    runner.tqdm.update(n=(runner.episodes - runner.tqdm_last_update))
                    runner.tqdm_last_update = runner.episodes
                    return inner_callback(runner, parallel)

            else:
                # Timestep-based tqdm
                self.tqdm = tqdm(
                    desc='Timesteps', total=self.num_timesteps, initial=self.timesteps,
                    postfix=dict(mean_reward='n/a')
                )
                self.tqdm_last_update = self.timesteps

                def tqdm_callback(runner, parallel):
                    # sum_timesteps_reward = sum(runner.timestep_rewards[num_mean_reward:])
                    # num_timesteps = min(num_mean_reward, runner.episode_timestep)
                    # mean_reward = sum_timesteps_reward / num_episodes
                    runner.tqdm.set_postfix(mean_reward='n/a')
                    runner.tqdm.update(n=(runner.timesteps - runner.tqdm_last_update))
                    runner.tqdm_last_update = runner.timesteps
                    return inner_callback(runner, parallel)

            self.callback = tqdm_callback

        # Evaluation
        if self.evaluation_environment is None:
            assert evaluation_callback is None
            assert self.save_best_agent is None
        else:
            if evaluation_callback is None:
                self.evaluation_callback = (lambda r: None)
            else:
                self.evaluation_callback = evaluation_callback
            if self.save_best_agent is not None:
                inner_evaluation_callback = self.evaluation_callback

                def mean_reward_callback(runner):
                    result = inner_evaluation_callback(runner)
                    if result is None:
                        return runner.evaluation_reward
                    else:
                        return result

                self.evaluation_callback = mean_reward_callback
                self.best_evaluation_score = None

        # Required if agent was previously stopped mid-episode
        self.agent.reset()

        # Reset environments and episode statistics
        for environment in self.environments:
            environment.start_reset()
        self.episode_reward = [0.0 for _ in self.environments]
        self.episode_timestep = [0 for _ in self.environments]
        self.episode_agent_second = [0.0 for _ in self.environments]
        episode_start = [time.time() for _ in self.environments]
        environments = list(self.environments)

        if self.evaluation_environment is not None:
            self.evaluation_environment.start_reset()
            self.evaluation_reward = 0.0
            self.evaluation_timestep = 0
            self.evaluation_agent_second = 0.0
            evaluation_start = time.time()
            environments.append(self.evaluation_environment)

        if self.sync_episodes:
            terminated = [False for _ in environments]

        # Runner loop
        while True:

            if not self.sync_timesteps:
                no_environment_ready = True

            # Parallel environments loop
            for parallel, environment in enumerate(environments):

                # Is evaluation environment?
                evaluation = (parallel == len(self.environments))

                if self.sync_episodes and terminated[parallel]:
                    # Continue if episode terminated
                    continue

                if self.sync_timesteps:
                    # Wait until environment is ready
                    while True:
                        observation = environment.retrieve_execute()
                        if observation is not None:
                            break
                        time.sleep(self.num_sleep_secs)

                else:
                    # Check whether environment is ready
                    observation = environment.retrieve_execute()
                    if observation is None:
                        continue
                    no_environment_ready = False

                states, terminal, reward = observation

                # Episode start or evaluation
                if terminal is None:
                    # Retrieve actions from agent
                    agent_start = time.time()
                    actions = self.agent.act(
                        states=states, parallel=(parallel - int(evaluation)), evaluation=evaluation
                    )

                    if evaluation:
                        self.evaluation_agent_second += time.time() - agent_start
                        self.evaluation_timestep += 1
                    else:
                        self.timesteps += 1
                        self.episode_agent_second[parallel] += time.time() - agent_start
                        self.episode_timestep[parallel] += 1

                    # Execute actions in environment
                    environment.start_execute(actions=actions)

                    continue

                elif isinstance(terminal, bool):
                    terminal = int(terminal)

                # Observe unless episode just started or evaluation
                # assert (terminal is None) == (self.episode_timestep[parallel] == 0)
                # if terminal is not None and not evaluation:
                if evaluation:
                    self.evaluation_reward += reward
                else:
                    agent_start = time.time()
                    updated = self.agent.observe(
                        terminal=terminal, reward=reward, parallel=parallel
                    )
                    self.updates += int(updated)
                    self.episode_agent_second[parallel] += time.time() - agent_start
                    self.episode_reward[parallel] += reward

                # # Update global timesteps/episodes/updates
                # self.global_timesteps = self.agent.timesteps
                # self.global_episodes = self.agent.episodes
                # self.global_updates = self.agent.updates

                # Callback plus experiment termination check
                if not evaluation and \
                        self.episode_timestep[parallel] % self.callback_timestep_frequency == 0 and \
                        not self.callback(self, parallel):
                    return

                if terminal > 0:
                    if evaluation:
                        # Update experiment statistics
                        self.evaluation_rewards.append(self.evaluation_reward)
                        self.evaluation_timesteps.append(self.evaluation_timestep)
                        self.evaluation_seconds.append(time.time() - evaluation_start)
                        self.evaluation_agent_seconds.append(self.evaluation_agent_second)

                        # Evaluation callback
                        if self.save_best_agent is not None:
                            evaluation_score = self.evaluation_callback(self)
                            assert isinstance(evaluation_score, float)
                            if self.best_evaluation_score is None:
                                self.best_evaluation_score = evaluation_score
                            elif evaluation_score > self.best_evaluation_score:
                                self.best_evaluation_score = evaluation_score
                                self.agent.save(
                                    directory=self.save_best_agent, filename='best-model',
                                    append_timestep=False
                                )
                        else:
                            self.evaluation_callback(self)

                    else:
                        # Increment episode counter (after calling callback)
                        self.episodes += 1

                        # Update experiment statistics
                        self.episode_rewards.append(self.episode_reward[parallel])
                        self.episode_timesteps.append(self.episode_timestep[parallel])
                        self.episode_seconds.append(time.time() - episode_start[parallel])
                        self.episode_agent_seconds.append(self.episode_agent_second[parallel])

                        # Callback
                        if self.episodes % self.callback_episode_frequency == 0 and \
                                not self.callback(self, parallel):
                            return

                # Terminate experiment if too long
                if self.timesteps >= self.num_timesteps:
                    return
                elif self.episodes >= self.num_episodes:
                    return
                elif self.updates >= self.num_updates:
                    return
                elif self.agent.should_stop():
                    return

                # Check whether episode terminated
                if terminal > 0:

                    if self.sync_episodes:
                        terminated[parallel] = True

                    if evaluation:
                        # Reset environment and episode statistics
                        environment.start_reset()
                        self.evaluation_reward = 0.0
                        self.evaluation_timestep = 0
                        self.evaluation_agent_second = 0.0
                        evaluation_start = time.time()

                    else:
                        # Reset environment and episode statistics
                        environment.start_reset()
                        self.episode_reward[parallel] = 0.0
                        self.episode_timestep[parallel] = 0
                        self.episode_agent_second[parallel] = 0.0
                        episode_start[parallel] = time.time()

                else:
                    # Retrieve actions from agent
                    agent_start = time.time()
                    actions = self.agent.act(
                        states=states, parallel=(parallel - int(evaluation)), evaluation=evaluation
                    )

                    if evaluation:
                        self.evaluation_agent_second += time.time() - agent_start
                        self.evaluation_timestep += 1
                    else:
                        self.timesteps += 1
                        self.episode_agent_second[parallel] += time.time() - agent_start
                        self.episode_timestep[parallel] += 1

                    # Execute actions in environment
                    environment.start_execute(actions=actions)

            if not self.sync_timesteps and no_environment_ready:
                # Sleep if no environment was ready
                time.sleep(self.num_sleep_secs)

            if self.sync_episodes and all(terminated):
                # Reset if all episodes terminated
                terminated = [False for _ in environments]
