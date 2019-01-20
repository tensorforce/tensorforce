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
        callback='tqdm', callback_episode_frequency=None, callback_timestep_frequency=None,
        # Save
        save_frequency=None, save_directory=None, save_append_timestep=False
    ):
        # Callback
        if callback_episode_frequency is None and callback_timestep_frequency is None:
            callback_episode_frequency = 1

        if callback == 'tqdm':
            # Tqdm callback
            assert callback_episode_frequency is None or callback_timestep_frequency is None
            from tqdm import tqdm

            if callback_episode_frequency is None:
                # Timestep-based tqdm
                assert num_timesteps is not None
                self.tqdm = tqdm(total=num_timesteps, initial=self.global_timestep)
                self.last_update = self.global_timestep

                def callback(runner, parallel):
                    runner.tqdm.update(n=(runner.global_timestep - runner.last_update))
                    runner.last_update = runner.global_timestep
                    return True

            else:
                # Episode-based tqdm
                assert num_episodes is not None
                self.tqdm = tqdm(total=num_episodes, initial=self.global_episode)
                self.last_update = self.global_episode

                def callback(runner, parallel):
                    runner.tqdm.update(n=(runner.global_episode - runner.last_update))
                    runner.last_update = runner.global_episode
                    return True

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
            # Save agent
            if save_frequency is not None and (self.episode % save_frequency) == 0 and \
                    self.episode > 0:
                self.agent.save_model(
                    directory=save_directory, append_timestep=save_append_timestep
                )

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
                if max_episode_timesteps is not None and \
                        self.episode_timestep[parallel] >= max_episode_timesteps:
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
                if (
                    callback_timestep_frequency is not None and
                    (self.episode_timestep[parallel] % callback_timestep_frequency) == 0 and
                    not callback(self, parallel)
                ):
                    return

                if terminal:
                    # Update experiment statistics
                    self.episode_rewards.append(self.episode_reward[parallel])
                    self.episode_timesteps.append(self.episode_timestep[parallel])
                    self.episode_times.append(time.time() - episode_start[parallel])

                    # Callback
                    if callback_episode_frequency is not None and \
                            (self.episode % callback_episode_frequency) == 0 and \
                            not callback(self, parallel):
                        return

                    # Increment episode counter (after calling callback)
                    self.episode += 1

                # Terminate experiment if too long
                if num_timesteps is not None and self.global_timestep >= num_timesteps:
                    return
                elif num_episodes is not None and self.global_episode >= num_episodes:
                    return
                elif self.agent.should_stop():
                    return

                # Check whether episode terminated
                if terminal:
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
