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

"""
Runner for non-realtime threaded execution of multiple agents.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import threading
from six.moves import xrange
from tensorforce.agents.agent import Agent
from tensorforce import TensorForceError


class ThreadedRunner(object):
    def __init__(self, agents, environments, repeat_actions=1, save_path=None, save_episodes=None):
        """
        Initialize a Runner object.

        Args:
            agent: `Agent` object containing the reinforcement learning agent
            environment: `../../environments/Environment` object containing
            repeat_actions:
            save_path:
            save_episodes:
        """
        if len(agents) != len(environments):
            raise TensorForceError("Each agent must have its own environment. Got {a} agents and {e} environments.".
                                   format(a=len(agents), e=len(environments)))
        self.agents = agents
        self.environments = environments
        self.repeat_actions = repeat_actions
        self.save_path = save_path
        self.save_episodes = save_episodes

    def _run_single(self, thread_id, agent, environment, repeat_actions=1, max_timesteps=-1, episode_finished=None):
        """
        The target function for a thread, runs an agent and environment until signaled to stop.
        Adds rewards to shared episode rewards list.

        Args:
            max_timesteps: Max timesteps in a given episode
            episode_finished: Optional termination condition, e.g. a particular mean reward threshold

        Returns:

        """
        episode = 1
        while not self.global_should_stop:
            state = environment.reset()
            agent.reset()
            episode_reward = 0

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

                if terminal or timestep == max_timesteps:
                    break

                if self.global_should_stop:
                    return

            #agent.observe_episode_reward(episode_reward)
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

    def run(self, episodes=-1, max_timesteps=-1, episode_finished=None, summary_report=None, summary_interval=0):
        # Save episode reward and length for statistics.
        self.episode_rewards = []
        self.episode_lengths = []

        self.global_step = 0
        self.global_episode = 1
        self.global_should_stop = False

        # Create threads
        threads = [threading.Thread(target=self._run_single, args=(t, self.agents[t], self.environments[t],),
                                    kwargs={"repeat_actions": self.repeat_actions,
                                            "max_timesteps": max_timesteps,
                                            "episode_finished": episode_finished})
                   for t in range(len(self.agents))]

        # Start threads
        self.start_time = time.time()
        [t.start() for t in threads]

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
    Worker Agent generator, recieves an Agent class and creates a Worker Agent class that inherits from that Agent.
    """

    class WorkerAgent(agent_class):
        """
        Worker agent receiving a shared model to avoid creating multiple models.
        """

        def __init__(self, states_spec, actions_spec, network_spec, model=None, **kwargs):
            self.network_spec = network_spec
            self.model = model

            super(WorkerAgent, self).__init__(
                states_spec,
                actions_spec,
                **kwargs
            )

        def initialize_model(self, states_spec, actions_spec):
            return self.model

    return WorkerAgent
