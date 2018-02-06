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


class BaseRunner(object):
    """
    Base class for all runner classes.
    Implements the `run` method.
    """
    def __init__(self, agent, environment, repeat_actions=1, history=None):
        """
        Args:
            agent (Agent): Agent object (or list of Agent objects) to use for the run.
            environment (Environment): Environment object (or list of Environment objects) to use for the run.
            repeat_actions (int): How many times the same given action will be repeated in subsequent calls to
                Environment's `execute` method. Rewards collected in these calls are accumulated and reported
                as a sum in the following call to Agent's `observe` method.
            history (dict): A dictionary containing an already run experiment's results. Keys should be:
                episode_rewards (list of rewards), episode_timesteps (lengths of episodes), episode_times (run-times)
        """
        self.agent = agent
        self.environment = environment
        self.repeat_actions = repeat_actions

        self.global_episode = None  # the global episode number (across all (parallel) agents)
        self.global_timestep = None  # the global time step (across all (parallel) agents)

        self.start_time = None  # TODO: is this necessary here? global start time (episode?, overall?)

        # lists of episode data (rewards, wall-times/timesteps)
        self.episode_rewards = None  # list of accumulated episode rewards
        self.episode_timesteps = None  # list of total timesteps taken in the episodes
        self.episode_times = None  # list of durations for the episodes

        self.reset(history)

    def reset(self, history=None):
        """
        Resets the Runner's internal stats counters.
        If history is empty, use default values in history.get().

        Args:
            history (dict): A dictionary containing an already run experiment's results. Keys should be:
                episode_rewards (list of rewards), episode_timesteps (lengths of episodes), episode_times (run-times)
        """
        if not history:
            history = dict()

        self.episode_rewards = history.get("episode_rewards", list())
        self.episode_timesteps = history.get("episode_timesteps", list())
        self.episode_times = history.get("episode_times", list())

    def close(self):
        """
        Should perform clean up operations on Runner's Agent(s) and Environment(s).
        """
        raise NotImplementedError

    def run(self, num_episodes, num_timesteps, max_episode_timesteps, deterministic, episode_finished, summary_report,
            summary_interval):
        """
        Executes this runner by starting to act (via Agent(s)) in the given Environment(s).
        Stops execution according to certain conditions (e.g. max. number of episodes, etc..).
        Calls callback functions after each episode and/or after some summary criteria are met.

        Args:
            num_episodes (int): Max. number of episodes to run globally in total (across all threads/workers).
            num_timesteps (int): Max. number of time steps to run globally in total (across all threads/workers)
            max_episode_timesteps (int): Max. number of timesteps per episode.
            deterministic (bool): Whether to use exploration when selecting actions.
            episode_finished (callable): A function to be called once an episodes has finished. Should take
                a BaseRunner object and some worker ID (e.g. thread-ID or task-ID). Can decide for itself
                every how many episodes it should report something and what to report.
            summary_report (callable): Deprecated; Function that could produce a summary over the training
                progress so far.
            summary_interval (int): Deprecated; The number of time steps to execute (globally)
                before summary_report is called.
        """
        raise NotImplementedError

    # keep backwards compatibility
    @property
    def episode(self):
        """
        Deprecated property `episode` -> global_episode.
        """
        return self.global_episode

    @property
    def timestep(self):
        """
        Deprecated property `timestep` -> global_timestep.
        """
        return self.global_timestep


