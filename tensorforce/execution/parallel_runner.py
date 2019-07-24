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

    def __init__(self, agent, environments, evaluation_environment=None):
        if not util.is_iterable(x=environments):
            raise TensorforceError.type(
                name='parallel-runner', argument='environments', value=environments
            )
        elif len(environments) == 0:
            raise TensorforceError.value(
                name='parallel-runner', argument='environments', value=environments
            )

        self.is_environment_external = tuple(
            isinstance(environment, Environment) for environment in environments
        )
        self.environments = tuple(
            Environment.create(environment=environment) for environment in environments
        )
        self.is_agent_external = isinstance(agent, Agent)
        self.agent = Agent.create(
            agent=agent, environment=self.environments[0], parallel_interactions=len(environments)
        )
        self.is_eval_environment_external = isinstance(evaluation_environment, Environment)
        if evaluation_environment is None:
            self.evaluation_environment = None
        else:
            self.evaluation_environment = Environment.create(environment=evaluation_environment)

        if len(environments) > agent.parallel_interactions:
            raise TensorforceError(message="Too many environments.")

        self.agent.initialize()
        self.global_episode = self.agent.episode
        self.global_timestep = self.agent.timestep
        self.episode_rewards = list()
        self.episode_timesteps = list()
        self.episode_times = list()

    def close(self):
        if hasattr(self, 'tqdm'):
            self.tqdm.close()
        if not self.is_agent_external:
            self.agent.close()
        for is_external, environment in zip(self.is_environment_external, self.environments):
            if not is_external:
                environment.close()
        if self.evaluation_environment is not None and not self.is_eval_environment_external:
            self.evaluation_environment.close()
        self.agent.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(
        self,
        # General
        num_episodes=None, num_timesteps=None, max_episode_timesteps=None, num_sleep_secs=0.01,
        sync_timesteps=False, sync_episodes=False,
        # Callback
        callback=None, callback_episode_frequency=None, callback_timestep_frequency=None,
        # Tqdm
        use_tqdm=True, mean_horizon=10,
        # Evaluation
        evaluation_callback=None, save_best_agent=False
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
        if max_episode_timesteps is None:
            self.max_episode_timesteps = float('inf')
        else:
            self.max_episode_timesteps = max_episode_timesteps
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

        # Tqdm
        if use_tqdm:
            if hasattr(self, 'tqdm'):
                self.tqdm.close()

            assert self.num_episodes != float('inf') or self.num_timesteps != float('inf')
            inner_callback = self.callback

            if self.num_episodes != float('inf'):
                # Episode-based tqdm (default option if both num_episodes and num_timesteps set)
                assert self.num_episodes != float('inf')
                self.tqdm = tqdm(
                    desc='Episodes', total=self.num_episodes, initial=self.global_episode,
                    postfix={
                        'reward': '{:.2f}'.format(0.0), 'ts/ep': str(0),
                        'sec/ep': '{:.2f}'.format(0.0), 'ms/ts': str(0)
                    }
                )
                self.tqdm_last_update = self.global_episode

                def tqdm_callback(runner, parallel):
                    mean_reward = float(np.mean(runner.episode_rewards[-mean_horizon:]))
                    mean_ts_per_ep = int(np.mean(runner.episode_timesteps[-mean_horizon:]))
                    mean_sec_per_ep = float(np.mean(runner.episode_times[-mean_horizon:]))
                    mean_ms_per_ts = mean_sec_per_ep * 1000.0 / mean_ts_per_ep
                    runner.tqdm.set_postfix({
                        'reward': '{:.2f}'.format(mean_reward), 'ts/ep': str(mean_ts_per_ep),
                        'sec/ep': '{:.2f}'.format(mean_sec_per_ep),
                        'ms/ts': '{:.1f}'.format(mean_ms_per_ts)
                    })
                    runner.tqdm.update(n=(runner.global_episode - runner.tqdm_last_update))
                    runner.tqdm_last_update = runner.global_episode
                    return inner_callback(runner, parallel)

            else:
                # Timestep-based tqdm
                self.tqdm = tqdm(
                    desc='Timesteps', total=self.num_timesteps, initial=self.global_timestep,
                    postfix=dict(mean_reward='n/a')
                )
                self.tqdm_last_update = self.global_timestep

                def tqdm_callback(runner, parallel):
                    # sum_timesteps_reward = sum(runner.timestep_rewards[num_mean_reward:])
                    # num_timesteps = min(num_mean_reward, runner.episode_timestep)
                    # mean_reward = sum_timesteps_reward / num_episodes
                    runner.tqdm.set_postfix(mean_reward='n/a')
                    runner.tqdm.update(n=(runner.global_timestep - runner.tqdm_last_update))
                    runner.tqdm_last_update = runner.global_timestep
                    return inner_callback(runner, parallel)

            self.callback = tqdm_callback

        # Evaluation
        if self.evaluation_environment is None:
            assert evaluation_callback is None
            assert not save_best_agent
        else:
            if evaluation_callback is None:
                self.evaluation_callback = (lambda r: None)
            else:
                self.evaluation_callback = evaluation_callback
            self.save_best_agent = save_best_agent
            if self.save_best_agent:
                inner_evaluation_callback = self.evaluation_callback

                def mean_reward_callback(runner):
                    result = inner_evaluation_callback(runner)
                    if result is None:
                        return runner.evaluation_reward
                    else:
                        return result

                self.evaluation_callback = mean_reward_callback
                self.best_evaluation_score = None

        # Reset agent
        self.agent.reset()

        # Episode counter
        self.episode = 1

        # Reset environments and episode statistics
        for environment in self.environments:
            environment.start_reset()
        self.episode_reward = [0.0 for _ in self.environments]
        self.episode_timestep = [0 for _ in self.environments]
        episode_start = [time.time() for _ in self.environments]
        environments = list(self.environments)

        if self.evaluation_environment is not None:
            self.evaluation_environment.start_reset()
            self.evaluation_reward = 0.0
            self.evaluation_timestep = 0
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
                        time.sleep(num_sleep_secs)

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
                    actions = self.agent.act(
                        states=states, parallel=(parallel - int(evaluation)), evaluation=evaluation
                    )

                    if evaluation:
                        self.evaluation_timestep += 1
                    else:
                        self.episode_timestep[parallel] += 1

                    # Execute actions in environment
                    environment.start_execute(actions=actions)

                    continue

                # Terminate episode if too long
                if evaluation:
                    if self.evaluation_timestep >= self.max_episode_timesteps:
                        terminal = True
                else:
                    if self.episode_timestep[parallel] >= self.max_episode_timesteps:
                        terminal = True

                # Observe unless episode just started or evaluation
                # assert (terminal is None) == (self.episode_timestep[parallel] == 0)
                # if terminal is not None and not evaluation:
                if evaluation:
                    self.evaluation_reward += reward
                else:
                    self.agent.observe(terminal=terminal, reward=reward, parallel=parallel)
                    self.episode_reward[parallel] += reward

                # Update global timestep/episode
                self.global_timestep = self.agent.timestep
                self.global_episode = self.agent.episode

                # Callback plus experiment termination check
                if not evaluation and \
                        self.episode_timestep[parallel] % self.callback_timestep_frequency == 0 and \
                        not self.callback(self, parallel):
                    return

                if not evaluation and terminal:
                    # Update experiment statistics
                    self.episode_rewards.append(self.episode_reward[parallel])
                    self.episode_timesteps.append(self.episode_timestep[parallel])
                    self.episode_times.append(time.time() - episode_start[parallel])

                    # Callback
                    if self.episode % self.callback_episode_frequency == 0 and \
                            not self.callback(self, parallel):
                        return

                # Terminate experiment if too long
                if self.global_timestep >= self.num_timesteps:
                    return
                elif self.global_episode >= self.num_episodes:
                    return
                elif self.agent.should_stop():
                    return

                # Check whether episode terminated
                if terminal:

                    if self.sync_episodes:
                        terminated[parallel] = True

                    if evaluation:
                        # Evaluation episode terminated
                        self.evaluation_time = time.time() - evaluation_start

                        # Evaluation callback
                        if self.save_best_agent:
                            evaluation_score = self.evaluation_callback(self)
                            assert isinstance(evaluation_score, float)
                            if self.best_evaluation_score is None:
                                self.best_evaluation_score = evaluation_score
                            elif evaluation_score > self.best_evaluation_score:
                                self.best_evaluation_score = evaluation_score
                                self.agent.save(filename='best-model', append_timestep=False)
                        else:
                            self.evaluation_callback(self)

                        # Reset environment and episode statistics
                        environment.start_reset()
                        self.evaluation_reward = 0.0
                        self.evaluation_timestep = 0
                        evaluation_start = time.time()

                    else:
                        # Increment episode counter (after calling callback)
                        self.episode += 1

                        # Reset environment and episode statistics
                        environment.start_reset()
                        self.episode_reward[parallel] = 0.0
                        self.episode_timestep[parallel] = 0
                        episode_start[parallel] = time.time()

                else:
                    # Retrieve actions from agent
                    actions = self.agent.act(
                        states=states, parallel=(parallel - int(evaluation)), evaluation=evaluation
                    )

                    if evaluation:
                        self.evaluation_timestep += 1
                    else:
                        self.episode_timestep[parallel] += 1

                    # Execute actions in environment
                    environment.start_execute(actions=actions)

            if not self.sync_timesteps and no_environment_ready:
                # Sleep if no environment was ready
                time.sleep(num_sleep_secs)

            if self.sync_episodes and all(terminated):
                # Reset if all episodes terminated
                terminated = [False for _ in environments]
