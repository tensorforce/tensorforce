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

import numpy as np

from tensorforce import TensorforceError, util
from tensorforce.agents import Agent


class ParallelRunner(object):

    def __init__(self, agent, environments):
        if not util.is_iterable(x=environments):
            raise TensorforceError.type(
                name='parallel-runner', argument='environments', value=environments
            )
        elif len(environments) == 0:
            raise TensorforceError.value(
                name='parallel-runner', argument='environments', value=environments
            )

        if not isinstance(agent, Agent):
            agent = Agent.from_spec(
                spec=agent, states=environments[0].states(), actions=environments[0].actions(),
                parallel_interactions=len(environments)
            )

        if len(environments) > agent.parallel_interactions:
            raise TensorforceError(message="Too many environments.")

        self.agent = agent
        self.environments = tuple(environments)

        self.agent.initialize()
        self.global_episode = self.agent.episode
        self.global_timestep = self.agent.timestep
        self.episode_rewards = list()
        self.episode_timesteps = list()
        self.episode_times = list()

    def close(self):
        if hasattr(self, 'tqdm'):
            self.tqdm.close()
        self.agent.close()
        for environment in self.environments:
            environment.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(
        self,
        # General
        num_episodes=None, num_timesteps=None, max_episode_timesteps=None, deterministic=False,
        num_sleep_secs=0.1,
        # Callback
        callback=None, callback_episode_frequency=None, callback_timestep_frequency=None,
        # Tqdm
        use_tqdm=True, num_mean_reward=100
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
        self.deterministic = deterministic
        self.num_sleep_secs = num_sleep_secs

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
            from tqdm import tqdm

            if hasattr(self, 'tqdm'):
                self.tqdm.close()

            assert self.num_episodes != float('inf') or self.num_timesteps != float('inf')
            inner_callback = self.callback

            if self.num_episodes != float('inf'):
                # Episode-based tqdm (default option if both num_episodes and num_timesteps set)
                assert self.num_episodes != float('inf')
                self.tqdm = tqdm(
                    desc='Episodes', total=self.num_episodes, initial=self.global_episode,
                    postfix=dict(mean_reward=0.0)
                )
                self.tqdm_last_update = self.global_episode

                def tqdm_callback(runner, parallel):
                    mean_reward = float(np.mean(runner.episode_rewards[-num_mean_reward:]))
                    runner.tqdm.set_postfix(mean_reward=mean_reward)
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

        # Reset agent
        self.agent.reset()

        # Episode counter
        self.episode = 1

        # Reset environments and episode statistics
        for environment in self.environments:
            environment.start_reset()
        self.episode_reward = [0 for _ in self.environments]
        self.episode_timestep = [0 for _ in self.environments]
        episode_start = [time.time() for _ in self.environments]

        # Runner loop
        while True:
            # Parallel environments loop
            no_environment_ready = True
            for parallel, environment in enumerate(self.environments):
                observation = environment.retrieve_execute()

                # Check whether environment is ready
                if observation is None:
                    continue

                no_environment_ready = False
                states, terminal, reward = observation

                if terminal is None:
                    # Retrieve actions from agent
                    actions = self.agent.act(
                        states=states, deterministic=deterministic, parallel=parallel
                    )
                    self.episode_timestep[parallel] += 1

                    # Execute actions in environment
                    environment.start_execute(actions=actions)
                    continue

                # Terminate episode if too long
                if self.episode_timestep[parallel] >= self.max_episode_timesteps:
                    terminal = True

                # Observe unless episode just started
                assert (terminal is None) == (self.episode_timestep[parallel] == 0)
                if terminal is not None:
                    self.agent.observe(terminal=terminal, reward=reward, parallel=parallel)
                    self.episode_reward[parallel] += reward

                # Update global timestep/episode
                self.global_timestep = self.agent.timestep
                self.global_episode = self.agent.episode

                # Callback plus experiment termination check
                if self.episode_timestep[parallel] % self.callback_timestep_frequency == 0 and \
                        not self.callback(self, parallel):
                    return

                if terminal:
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
                    # Increment episode counter (after calling callback)
                    self.episode += 1

                    # Reset environment and episode statistics
                    environment.start_reset()
                    self.episode_reward[parallel] = 0
                    self.episode_timestep[parallel] = 0
                    episode_start[parallel] = time.time()

                else:
                    # Retrieve actions from agent
                    actions = self.agent.act(
                        states=states, deterministic=deterministic, parallel=parallel
                    )
                    self.episode_timestep[parallel] += 1

                    # Execute actions in environment
                    environment.start_execute(actions=actions)

            # Sleep if no environment was ready
            if no_environment_ready:
                time.sleep(num_sleep_secs)
