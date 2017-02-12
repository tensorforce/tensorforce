"""
Queued thread runner for real time environment execution.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from threading import Thread
from six.moves import xrange
import six.moves.queue as queue
from tensorforce.util.experiment_util import repeat_action


class ThreadRunner(Thread):

    def __init__(self, agent, environment, episodes, max_timesteps, preprocessor=None, repeat_actions=1):
        super(ThreadRunner, self).__init__()
        self.experience_queue = queue.Queue(10)

        self.agent = agent
        self.environment = environment
        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions
        self.episodes = episodes
        self.max_timesteps = max_timesteps

        self.save_model_path = None
        self.save_model_episodes = 0

    def run(self):
        """
        Starts threaded execution of environment execution.
        :return:
        """
        executor = self.execute()

        while True:
            self.experience_queue.put(next(executor), timeout=600.0)

    def execute(self):
        """
        Queued thread executor.
        """

        self.episode_rewards = []
        self.agent.model.initialize()

        # TODO
        # Currently update is hidden due to API
        # Need to be able to control this from outside
        # for asynchronous updates

        for self.episode in xrange(self.episodes):
            state = self.environment.reset()
            episode_reward = 0

            for self.timestep in xrange(self.max_timesteps):
                if self.preprocessor:
                    processed_state = self.preprocessor.process(state)
                else:
                    processed_state = state

                action = self.agent.get_action(processed_state, self.episode)
                result = repeat_action(self.environment, action, self.repeat_actions)

                episode_reward += result['reward']
                self.agent.add_observation(processed_state, action, result['reward'], result['terminal_state'])

                state = result['state']

                if result['terminal_state']:
                    break

            self.episode_rewards.append(episode_reward)

    def try_update(self):
        pass


