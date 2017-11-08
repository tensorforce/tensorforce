# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
from six.moves import xrange


class Runner(object):
    """
    Simple runner for non-realtime single-process execution.
    """

    def __init__(self, agent, environment, repeat_actions=1, history=None):
        """
        Initialize a Runner object.

        Args:
            agent:
            environment:
            repeat_actions:
        """
        self.agent = agent
        self.environment = environment
        self.repeat_actions = repeat_actions

        self.reset(history)

    def reset(self, history=None):
        # If history is empty, use default values in history.get().
        if not history:
            history = dict()

        self.episode_rewards = history.get('episode_rewards', list())
        self.episode_timesteps = history.get('episode_timesteps', list())
        self.episode_times = history.get('episode_times', list())

    def run(
        self,
        timesteps=None,
        episodes=None,
        max_episode_timesteps=None,
        deterministic=False,
        episode_finished=None
    ):
        """
        Runs the agent on the environment.

        Args:
            timesteps: Number of timesteps
            episodes: Number of episodes
            max_episode_timesteps: Max number of timesteps per episode
            deterministic: Deterministic flag
            episode_finished: Function handler taking a `Runner` argument and returning a boolean indicating
                whether to continue execution. For instance, useful for reporting intermediate performance or
                integrating termination conditions.
        """

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()

        self.agent.reset()

        self.episode = self.agent.episode
        if episodes is not None:
            episodes += self.agent.episode

        self.timestep = self.agent.timestep
        if timesteps is not None:
            timesteps += self.agent.timestep

        while True:
            episode_start_time = time.time()

            self.agent.reset()
            state = self.environment.reset()
            episode_reward = 0
            self.episode_timestep = 0

            while True:
                action = self.agent.act(states=state, deterministic=deterministic)

                if self.repeat_actions > 1:
                    reward = 0
                    for repeat in xrange(self.repeat_actions):
                        state, terminal, step_reward = self.environment.execute(actions=action)
                        reward += step_reward
                        if terminal:
                            break
                else:
                    state, terminal, reward = self.environment.execute(actions=action)

                if max_episode_timesteps is not None and self.episode_timestep >= max_episode_timesteps:
                    terminal = True

                self.agent.observe(terminal=terminal, reward=reward)

                self.episode_timestep += 1
                self.timestep += 1
                episode_reward += reward

                if terminal or self.agent.should_stop():  # TODO: should_stop also termina?
                    break

            time_passed = time.time() - episode_start_time

            self.episode_rewards.append(episode_reward)
            self.episode_timesteps.append(self.episode_timestep)
            self.episode_times.append(time_passed)

            self.episode += 1

            if episode_finished and not episode_finished(self) or \
                    (episodes is not None and self.agent.episode >= episodes) or \
                    (timesteps is not None and self.agent.timestep >= timesteps) or \
                    self.agent.should_stop():
                # agent.episode / agent.timestep are globally updated
                break

        self.agent.close()
        self.environment.close()
