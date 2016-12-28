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

from threading import Event, Thread


class AsyncRunner(Runner):

    def __init__(self, config, agent, environment, state_wrapper=None):
        super().__init__(config, agent, environment, state_wrapper=state_wrapper)
        self.n_runners = self.config.n_runners

    def run(self, episode_finished):
        self.total_states = 0
        self.episode_rewards = []

        runners = []
        threads = []
        events = []
        continue_execution = True
        for _ in range(self.n_runners):
            event = Event()
            events.append(events)

            def episode_finished(r):
                event.clear()
                event.wait()
                return continue_execution

            runner = Runner(self.config, deepcopy(self.agent), deepcopy(self.environment), state_wrapper=self.state_wrapper)
            runners.append(runner)
            thread = Thread(target=runner.run, kwargs={'episode_finished': episode_finished})
            threads.append(thread)
            thread.start()
        self.episode = 0
        loop = True
        while loop:
            for event, runner in zip(events, runners):
                if not event.is_set():
                    self.episode += 1
                    self.episode_rewards.append(runner.episode_rewards[-1])
                    # perform async update of parameters
                    # if T mod Itarget == 0:
                    #     update target network
                    # clear gradient
                    # sync parameters
                    if self.episode >= self.episodes or (episode_finished and not episode_finished(self)):
                        loop = False
                        break
                    event.set()
            self.total_states = sum(runner.total_states for runner in runners)

        continue_execution = False
        stopped = 0
        while stopped < self.n_runners:
            for event, runner in zip(events, runners):
                if not event.is_set():
                    event.set()
