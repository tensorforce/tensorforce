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


class Runner(object):
    """
    Simple runner for non-realtime single-process execution.
    """

    def __init__(self, agent, environment, parallel_id=0):
        """
        Initialize a single Runner object (one Agent/one Environment).

        Args:
            id_ (int): The ID of this Runner (for distributed TF runs).
        """
        self.agent = agent
        self.environment = environment
        self.parallel_id = parallel_id

        self.agent.initialize()
        self.global_episode = self.agent.episode
        self.global_timestep = self.agent.timestep
        self.episode_rewards = list()
        self.episode_timesteps = list()
        self.episode_times = list()

    def close(self):
        self.agent.close()
        self.environment.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(
        self,
        # General
        num_episodes=None, num_timesteps=None, max_episode_timesteps=None, deterministic=False,
        num_repeat_actions=1, num_sleep_secs=0,
        # Callback
        callback='tqdm', callback_episode_frequency=1, callback_timestep_frequency=None,
        # Evaluation
        evaluation_callback=None, evaluation_frequency=None, max_evaluation_timesteps=None,
        num_evaluation_iterations=1,
        # Save
        save_frequency=None, save_directory=None, save_append_timestep=False
    ):
        self.num_episodes = num_episodes
        self.num_timesteps = num_timesteps
        self.deterministic = deterministic
        self.num_repeat_actions = num_repeat_actions
        self.num_sleep_secs = num_sleep_secs

        # Callback
        assert not callback_episode_frequency or num_episodes is not None
        assert not callback_timestep_frequency or num_timesteps is not None
        assert callback is None or \
            (callback_episode_frequency is None) != (callback_timestep_frequency is None)
        self.callback_episode_frequency = callback_episode_frequency
        self.callback_timestep_frequency = callback_timestep_frequency

        if callback == 'tqdm':
            # Tqdm callback
            from tqdm import tqdm
            if self.callback_episode_frequency is None:
                # Timestep-based tqdm
                self.tqdm = tqdm(total=num_timesteps, initial=self.global_timestep)
                self.last_update = self.global_timestep

                def callback(runner):
                    runner.tqdm.update(n=(runner.global_timestep - runner.last_update))
                    runner.last_update = runner.global_timestep
                    return True

            else:
                # Episode-based tqdm
                self.tqdm = tqdm(total=num_episodes, initial=self.global_episode)
                self.last_update = self.global_episode

                def callback(runner):
                    runner.tqdm.update(n=(runner.global_episode - runner.last_update))
                    runner.last_update = runner.global_episode
                    return True

        self.callback = callback

        # Episode counter
        self.episode = 1

        # Episode loop
        while True:
            # Save agent
            if save_frequency is not None and (self.episode % save_frequency) == 0 and \
                    self.episode > 0:
                self.agent.save_model(
                    directory=save_directory, append_timestep=save_append_timestep
                )

            # Run evaluation
            if evaluation_callback is not None and (self.episode % evaluation_frequency) == 0 \
                    and self.episode > 0:
                self.evaluation_rewards = list()
                self.evaluation_timesteps = list()
                self.evaluation_times = list()
                for _ in range(num_evaluation_iterations):
                    self.run_episode(max_timesteps=max_evaluation_timesteps, evaluation=True)
                    self.evaluation_rewards.append(self.episode_reward)
                    self.evaluation_timesteps.append(self.episode_timestep)
                    self.evaluation_times.append(self.episode_time)
                evaluation_callback(self)

            # Run episode
            if not self.run_episode(max_timesteps=max_episode_timesteps, evaluation=False):
                return

            # Update experiment statistics
            self.episode += 1
            self.episode_rewards.append(self.episode_reward)
            self.episode_timesteps.append(self.episode_timestep)
            self.episode_times.append(self.episode_time)

            # Update global timestep/episode
            self.global_timestep = self.agent.timestep
            self.global_episode = self.agent.episode

            # Callback plus experiment termination check
            if num_episodes is not None and callback is not None and \
                    (self.episode % self.callback_episode_frequency) == 0 and \
                    not self.callback(self):
                return
            elif self.agent.should_stop():
                return

            # Terminate experiment if too long
            if num_timesteps is not None and self.global_timestep >= num_timesteps:
                return
            elif num_episodes is not None and self.global_episode >= num_episodes:
                return

    def run_episode(self, max_timesteps, evaluation):
        # Episode statistics
        self.episode_timestep = 0
        self.episode_reward = 0
        self.episode_start = time.time()

        # Start environment episode
        states = self.environment.reset()

        # Timestep loop
        while True:
            # Retrieve actions from agent
            actions = self.agent.act(
                states=states, deterministic=(self.deterministic or evaluation),
                parallel=self.parallel_id
            )

            # Execute actions in environment (optional multiple executions)
            reward = 0
            for _ in range(self.num_repeat_actions):
                states, terminal, step_reward = self.environment.execute(actions=actions)
                reward += step_reward
                if terminal:
                    break

            # Update episode statistics
            self.episode_timestep += 1
            self.episode_reward += reward

            # Terminate episode if too long
            if max_timesteps is not None and self.current_timestep >= max_timesteps:
                terminal = True

            # Observe if not evaluation
            if not evaluation:
                self.agent.observe(terminal=terminal, reward=reward, parallel=self.parallel_id)

            # Episode termination check
            if terminal:
                break

            # Sleep
            if self.num_sleep_secs > 0:
                time.sleep(secs=self.num_sleep_secs)

            if evaluation:
                continue

            # Update global timestep/episode
            self.global_timestep = self.agent.timestep
            self.global_episode = self.agent.episode

            # Callback plus experiment termination check
            if self.num_timesteps is not None and self.callback is not None and \
                    (self.episode_timestep % self.callback_timestep_frequency) == 0 and \
                    not self.callback(self):
                return False
            elif self.agent.should_stop():
                return False

            # Terminate experiment if too long
            if self.num_timesteps is not None and self.global_timestep >= self.num_timesteps:
                return
            elif self.num_episodes is not None and self.global_episode >= self.num_episodes:
                return

        # Episode statistics
        self.episode_end = time.time()
        self.episode_time = self.episode_end - self.episode_start

        return True
