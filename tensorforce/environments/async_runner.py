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

from tensorforce.environments.runner import Runner


class AsyncRunner(Runner):

    default_config = {
        'episodes': 10000,
        'max_timesteps': 2000,
        'repeat_actions': 4,
        'n_runners': 2
    }

    def __init__(self, config, agent, environment, preprocessor=None):
        super(AsynRunner, self).__init__(config, agent, environment, preprocessor=preprocessor)
        self.n_runners = self.config.n_runners
        self.episodes = 3

    def run(self, episode_finished=None):
        self.total_states = 0
        self.episode_rewards = []

        runners = []
        threads = []
        conditions = []
        continue_execution = True
        for _ in range(self.n_runners):
            condition = Condition()
            conditions.append(condition)

            runner = Runner(self.config, self.agent, self.environment, preprocessor=self.preprocessor)  # deepcopy
            runners.append(runner)

            def runner_episode_finished(r):
                condition.acquire()
                condition.wait()
                return continue_execution

            thread = Thread(target=runner.run, kwargs={'episode_finished': runner_episode_finished})
            threads.append(thread)
            thread.start()

        self.episode = 0
        loop = True
        while loop:
            for condition, runner in zip(conditions, runners):
                if condition._waiters:
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
                    if self.episode >= self.episodes or (episode_finished and not episode_finished(self)):
                        loop = False
                        break
            self.total_states = sum(runner.total_states for runner in runners)

        continue_execution = False
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
