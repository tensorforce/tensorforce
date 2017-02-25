# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Simple runner for non-realtime single process execution, appropriate for
OpenAI gym.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange

from tensorforce.util.experiment_util import repeat_action


class Runner(object):

    def __init__(self, agent, environment, preprocessor=None, repeat_actions=1):
        self.agent = agent
        self.environment = environment
        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions
        self.save_model_path = None
        self.save_model_episodes = 0

    def save_model(self, path, num_episodes):
        self.save_model_path = path
        self.save_model_episodes = num_episodes

    def run(self, episodes, max_timesteps, episode_finished=None):
        self.episode_rewards = []  # save all episode rewards for statistics

        self.agent.setup()

        for self.episode in xrange(episodes):
            state = self.environment.reset()
            episode_reward = 0

            for self.timestep in xrange(max_timesteps):
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

            if self.save_model_path and self.save_model_episodes > 0 and self.episode % self.save_model_episodes == 0:
                print("Saving agent after episode {}".format(self.episode))
                self.agent.save_model(self.save_model_path)

            if episode_finished and not episode_finished(self):
                return
