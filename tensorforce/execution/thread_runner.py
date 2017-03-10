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
Queued thread runner for real time environment execution.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from threading import Thread

from six.moves import xrange
import six.moves.queue as queue

from tensorforce.agents.distributed_agent import Experience
from tensorforce.util.experiment_util import repeat_action


class ThreadRunner(Thread):
    def __init__(self, agent, environment, max_episode_steps, local_steps, preprocessor=None,
                 repeat_actions=1):
        super(ThreadRunner, self).__init__()
        self.daemon = True
        self.experience_queue = queue.Queue(5)

        self.agent = agent
        self.environment = environment
        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions
        self.max_episode_steps = max_episode_steps
        self.local_steps = local_steps

        self.save_model_path = None
        self.save_model_episodes = 0

    def start_thread(self, session):
        """
        Starts threaded execution of environment execution.
        """
        self.agent.set_session(session)

        self.start()

    def run(self):
        print('Starting thread runner..')
        executor = self.execute()

        while True:
            self.experience_queue.put(next(executor), timeout=600.0)

    def execute(self):
        """
        Queued thread executor.
        """
        self.episode_rewards = []
        state = self.environment.reset()

        # For episode dependent exploration
        current_episode = 1
        current_episode_step = 0
        current_episode_rewards = 0

        while True:

            # Tell agent to create a new experience fragment
            # TODO Michael -> I don't like external experience management. Refactor
            experience = Experience(self.agent.continuous)

            for _ in xrange(self.local_steps):
                if self.preprocessor:
                    processed_state = self.preprocessor.process(state)
                else:
                    processed_state = state

                current_episode_step += 1
                action = self.agent.get_action(processed_state, current_episode, experience=experience)
                result = repeat_action(self.environment, action, self.repeat_actions)

                experience.add_observation(processed_state, action, result['reward'], result['terminal_state'])

                current_episode_rewards += result['reward']
                state = result['state']

                if result['terminal_state'] or current_episode_step >= self.max_episode_steps:
                    print('Episode finished after ' + str(current_episode_step) + ' steps, episode reward= ' + str(
                        current_episode_rewards))

                    self.episode_rewards.append(current_episode_rewards)
                    state = self.environment.reset()
                    current_episode += 1

                    current_episode_step = 0
                    current_episode_rewards = 0

                    break

            # TODO argument to be made to move the queue into the agent
            yield experience

    def update(self):
        self.agent.sync()

        # We yield the current episode fragment
        experience = self.experience_queue.get(timeout=600.0)

        # Append to current episode in agent
        self.agent.extend(experience)

        # Turn the openai starter agent logic on its head so we can
        # actively call update and don't break encapsulation of our model logic ->
        # model does not know or care about environment
        while not self.agent.current_episode['terminated']:
            try:
                # This experience fragment is part of a single episode in a game
                # the agent should know how to concatenate this
                # experience.extend(self.experience_queue.get_nowait())
                self.agent.extend(self.experience_queue.get_nowait())

            except queue.Empty:
                break

        # Delegate update to distributed model, separate queue runner and update
        # agent manages experience fragments internally -> no need to pass here
        self.agent.update()
