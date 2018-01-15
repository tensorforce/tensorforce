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
import threading
from six.moves import xrange
import warnings

from tensorforce import TensorForceError


class ThreadedRunner(object):
    """
    Runner for non-realtime threaded execution of multiple agents.
    """

    def __init__(self, agents, environments, repeat_actions=1, save_path=None, save_episodes=None):
        """
        Initialize a ThreadedRunner object.

        Args:
            agents (List[Agent]): List of Agent objects to use (one on each thread)
            environments (List[Environment]): List of Environment objects to use (one for each agent)
            repeat_actions (int): How many times the same given action will be repeated in subsequent calls to
                Environment's `execute` method. Rewards collected in these calls are accumulated and reported
                as a sum in the following call to Agent's `observe` method.
            save_path (str): Path where to save the shared model.
            save_episodes (int): Every how many (global) episodes do we save the shared model?
        """
        if len(agents) != len(environments):
            raise TensorForceError("Each agent must have its own environment. Got {a} agents and {e} environments.".
                                   format(a=len(agents), e=len(environments)))
        self.agents = agents
        self.environments = environments
        self.repeat_actions = repeat_actions
        self.save_path = save_path
        self.save_episodes = save_episodes

        # init some stats for the parallel runs
        self.episode_rewards = None  # global episode-rewards collected by the different workers
        self.episode_lengths = None  # global episode-lengths collected by the different workers
        self.start_time = None  # start time of a run (with many worker threads)
        self.global_step = None  # global step counter (sum over all workers)
        self.global_episode = None  # global episode counter (sum over all workers)
        self.global_should_stop = False  # global stop-condition flag that each worker abides to (aborts if True)

    def _run_single(self, thread_id, agent, environment, repeat_actions=1, max_episode_timesteps=-1,
                    episode_finished=None):
        """
        The target function for a thread, runs an agent and environment until signaled to stop.
        Adds rewards to shared episode rewards list.

        Args:
            max_episode_timesteps (int): Max. number of timesteps per episode. Use -1 or 0 for non-limited episodes.
            episode_finished (callable): Function called after each episode that takes an episode summary spec and
                returns False, if this single run should terminate after this episode.
                Can be used e.g. to set a particular mean reward threshold.
        """
        episode = 1
        # Run this single worker (episodes) as long as global count thresholds have not been reached.
        while not self.global_should_stop:
            state = environment.reset()
            agent.reset()
            episode_reward = 0

            # timestep (within episode) loop
            timestep = 0
            while True:
                action = agent.act(states=state)
                if repeat_actions > 1:
                    reward = 0
                    for repeat in xrange(repeat_actions):
                        state, terminal, step_reward = environment.execute(actions=action)
                        reward += step_reward
                        if terminal:
                            break
                else:
                    state, terminal, reward = environment.execute(actions=action)

                agent.observe(reward=reward, terminal=terminal)

                timestep += 1
                self.global_step += 1
                episode_reward += reward

                if terminal or timestep == max_episode_timesteps:
                    break

                # abort the episode (discard its results) when global says so
                if self.global_should_stop:
                    return

            #agent.observe_episode_reward(episode_reward)
            # TODO: Could cause a race condition where order in episode_rewards won't match order in episode_lengths.
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(timestep)

            summary_data = {
                "thread_id": thread_id,
                "episode": episode,
                "timestep": timestep,
                "episode_reward": episode_reward
            }
            if episode_finished and not episode_finished(summary_data):
                return

            episode += 1
            self.global_episode += 1

    def run(self, episodes=-1, max_episode_timesteps=-1, episode_finished=None, summary_report=None,
            summary_interval=0, max_timesteps=None):
        """

        Args:
            episodes (List[Episode]):
            max_episode_timesteps (int): Max. number of timesteps per episode.
            episode_finished (callable):
            summary_report (callable): Function that produces a tensorboard summary update.
            summary_interval (int):
            max_timesteps (int): Deprecated; see max_episode_timesteps
        """

        # Renamed max_timesteps into max_episode_timesteps to match single Runner's signature (fully backw. compatible).
        if max_timesteps is not None:
            max_episode_timesteps = max_timesteps
            warnings.warn("WARNING: `max_timesteps` parameter is deprecated, use `max_episode_timesteps` instead.",
                          category=DeprecationWarning)
        assert isinstance(max_episode_timesteps, int)

        # Save episode rewards and lengths for statistics.
        self.episode_rewards = []
        self.episode_lengths = []

        # Reset counts/stop-condition for this run.
        self.global_step = 0
        self.global_episode = 1
        self.global_should_stop = False

        # Create threads
        threads = [threading.Thread(target=self._run_single, args=(t, self.agents[t], self.environments[t],),
                                    kwargs={"repeat_actions": self.repeat_actions,
                                            "max_episode_timesteps": max_episode_timesteps,
                                            "episode_finished": episode_finished})
                   for t in range(len(self.agents))]

        # Start threads
        self.start_time = time.time()
        [t.start() for t in threads]

        # Stay idle until killed by SIGINT or a global stop condition is met.
        try:
            next_summary = 0
            next_save = 0
            while self.global_episode < episodes or episodes == -1:
                if self.global_episode > next_summary:
                    summary_report(self)
                    next_summary += summary_interval
                if self.save_path and self.save_episodes is not None and self.global_episode > next_save:
                    print("Saving agent after episode {}".format(self.global_episode))
                    self.agents[0].save_model(self.save_path)
                    next_save += self.save_episodes
                time.sleep(1)
        except KeyboardInterrupt:
            print('Keyboard interrupt, sending stop command to threads')

        self.global_should_stop = True

        # Join threads
        [t.join() for t in threads]
        print('All threads stopped')


def WorkerAgentGenerator(agent_class):
    """
    Worker Agent generator, receives an Agent class and creates a Worker Agent class that inherits from that Agent.
    """

    class WorkerAgent(agent_class):
        """
        Worker agent receiving a shared model to avoid creating multiple models.
        """

        def __init__(self, model=None, **kwargs):
            # set our model externally
            self.model = model
            # call super c'tor (which will call initialize_model and assing self.model to the return value)
            super(WorkerAgent, self).__init__(**kwargs)

        def initialize_model(self):
            # return our model (already given and initialized)
            return self.model

    return WorkerAgent

