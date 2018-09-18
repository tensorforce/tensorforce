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

from tensorforce.execution.base_runner import BaseRunner

import time
from six.moves import xrange
import warnings
from inspect import getargspec
from tqdm import tqdm

class ParallelRunner(BaseRunner):
    """
    Simple runner for non-realtime single-process execution.
    """

    def __init__(self, agent, environment, repeat_actions=1, history=None, id_=0):
        """
        Initialize a single Runner object (one Agent/one Environment).

        Args:
            id_ (int): The ID of this Runner (for distributed TF runs).
        """
        super(ParallelRunner, self).__init__(agent, environment, repeat_actions, history)

        self.id = id_  # the worker's ID in a distributed run (default=0)
        self.current_timestep = None  # the time step in the current episode
        self.episode_actions = []
        self.num_parallel = self.agent.execution['num_parallel']
        print('ParallelRunner with {} parallel buffers.'.format(self.num_parallel))

    def close(self):
        self.agent.close()
        self.environment.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(self, num_timesteps=None, num_episodes=None, max_episode_timesteps=None, deterministic=False, episode_finished=None, summary_report=None, summary_interval=None, timesteps=None, episodes=None, testing=False, sleep=None
            ):
        """
        Args:
            timesteps (int): Deprecated; see num_timesteps.
            episodes (int): Deprecated; see num_episodes.
        """

        # deprecation warnings
        if timesteps is not None:
            num_timesteps = timesteps
            warnings.warn("WARNING: `timesteps` parameter is deprecated, use `num_timesteps` instead.",
                          category=DeprecationWarning)
        if episodes is not None:
            num_episodes = episodes
            warnings.warn("WARNING: `episodes` parameter is deprecated, use `num_episodes` instead.",
                          category=DeprecationWarning)

        # figure out whether we are using the deprecated way of "episode_finished" reporting
        old_episode_finished = False
        if episode_finished is not None and len(getargspec(episode_finished).args) == 1:
            old_episode_finished = True

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()

        self.agent.reset()

        if num_episodes is not None:
            num_episodes += self.agent.episode

        if num_timesteps is not None:
            num_timesteps += self.agent.timestep

        # add progress bar
        with tqdm(total=num_episodes) as pbar:
            # episode loop
            index = 0
            while True:
                episode_start_time = time.time()
                state = self.environment.reset()
                self.agent.reset()

                # Update global counters.
                self.global_episode = self.agent.episode  # global value (across all agents)
                self.global_timestep = self.agent.timestep  # global value (across all agents)

                episode_reward = 0
                self.current_timestep = 0

                # time step (within episode) loop
                while True:
                    action = self.agent.act(states=state, deterministic=deterministic, index=index)

                    reward = 0
                    for _ in xrange(self.repeat_actions):
                        state, terminal, step_reward = self.environment.execute(action=action)
                        reward += step_reward
                        if terminal:
                            break

                    if max_episode_timesteps is not None and self.current_timestep >= max_episode_timesteps:
                        terminal = True

                    if not testing:
                        self.agent.observe(terminal=terminal, reward=reward, index=index)

                    self.global_timestep += 1
                    self.current_timestep += 1
                    episode_reward += reward

                    if terminal or self.agent.should_stop():  # TODO: should_stop also terminate?
                        break

                    if sleep is not None:
                        time.sleep(sleep)

                index = (index + 1) % self.num_parallel

                # Update our episode stats.
                time_passed = time.time() - episode_start_time
                self.episode_rewards.append(episode_reward)
                self.episode_timesteps.append(self.current_timestep)
                self.episode_times.append(time_passed)
                self.episode_actions.append(self.environment.conv_action)

                self.global_episode += 1
                pbar.update(1)

                # Check, whether we should stop this run.
                if episode_finished is not None:
                    # deprecated way (passing in only runner object):
                    if old_episode_finished:
                        if not episode_finished(self):
                            break
                    # new unified way (passing in BaseRunner AND some worker ID):
                    elif not episode_finished(self, self.id):
                        break
                if (num_episodes is not None and self.global_episode >= num_episodes) or \
                        (num_timesteps is not None and self.global_timestep >= num_timesteps) or \
                        self.agent.should_stop():
                    break
            pbar.update(num_episodes - self.global_episode)
