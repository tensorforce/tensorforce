# Copyright 2016 reinforce.io. All Rights Reserved.
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

from copy import deepcopy
from threading import Condition, Thread

from tensorforce.runner import Runner


class AsyncRunner(Runner):

    def __init__(self, agents, environments, preprocessor=None, repeat_actions=1):
        super(AsyncRunner, self).__init__(agents[0], environments[0], preprocessor=preprocessor, repeat_actions=repeat_actions)
        self.agents = agents[1:]
        self.environments = environments[1:]
        self.continue_execution = False

    def get_episode_finished_handler(self, condition):
        def episode_finished(runner):
            condition.acquire()
            condition.wait()
            return self.continue_execution
        return episode_finished

    def run(self, episodes, max_timesteps, episode_finished=None):
        self.total_states = 0
        self.episode_rewards = []
        self.continue_execution = True

        runners = []
        threads = []
        conditions = []
        for agent, environment in zip(self.agents, self.environments):
            condition = Condition()
            conditions.append(condition)

            runner = Runner(agent, environment, preprocessor=self.preprocessor, repeat_actions=self.repeat_actions)  # deepcopy?
            runners.append(runner)

            thread = Thread(target=runner.run, args=(episodes, max_timesteps), kwargs={'episode_finished': self.get_episode_finished_handler(condition)})
            threads.append(thread)
            thread.start()

        self.episode = 0
        loop = True
        while loop:
            for condition, runner in zip(conditions, runners):
                if condition._waiters:
                    self.timestep = runner.timestep
                    self.episode += 1
                    self.episode_rewards.append(runner.episode_rewards[-1])
                    # perform async update of parameters
                    # if T mod Itarget == 0:
                    #     update target network
                    # clear gradient
                    # sync parameters
                    condition.acquire()
                    condition.notify()
                    condition.release()
                    if self.episode >= episodes or (episode_finished and not episode_finished(self)):
                        loop = False
                        break
            self.total_states = sum(runner.total_states for runner in runners)

        self.continue_execution = False
        stopped = 0
        while stopped < self.n_runners:
            for condition, thread in zip(conditions, threads):
                if condition._waiters:
                    condition.acquire()
                    condition.notify()
                    condition.release()
                    conditions.remove(condition)
                if not thread.is_alive():
                    threads.remove(thread)
                    stopped += 1
