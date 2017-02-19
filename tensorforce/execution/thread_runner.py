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

    def __init__(self, agent, environment, episodes, local_steps, preprocessor=None, repeat_actions=1):
        super(ThreadRunner, self).__init__()
        self.experience_queue = queue.Queue(10)

        self.agent = agent
        self.environment = environment
        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions
        self.episodes = episodes
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
        executor = self.execute()

        while True:
            self.experience_queue.put(next(executor), timeout=600.0)

    def execute(self):
        """
        Queued thread executor.
        """
        self.episode_rewards = []
        self.agent.initialize()

        # TODO
        # Currently update is hidden due to API
        # Need to be able to control this from outside
        # for asynchronous updates

        for self.episode in xrange(self.episodes):
            state = self.environment.reset()
            episode_reward = 0

            for self.timestep in xrange(self.local_steps):
                if self.preprocessor:
                    processed_state = self.preprocessor.process(state)
                else:
                    processed_state = state

                action = self.agent.get_action(processed_state, self.episode)
                result = repeat_action(self.environment, action, self.repeat_actions)
                self.agent.add_observation(processed_state, action, result['reward'], result['terminal_state'])

                episode_reward += result['reward']

                state = result['state']

                if result['terminal_state']:
                    break

            self.episode_rewards.append(episode_reward)

        # Let agent manage experience collection in internal batch -> possibly move the yield
        # into agent
        yield self.agent.batch

    def update(self):
        batch = self.experience_queue.get(timeout=600.0)

        # Turn the openai starter agent logic on its head so we can
        # actively call update and don't break encapsulation of our model logic ->
        # model does not know or care about environment
        while not batch.terminal:
            try:
                batch.extend(self.experience_queue.queue.get_nowait())
            except queue.Empty:
                break

        # Delegate update to distributed model, separate queue runner and update
        self.agent.update(batch.current_batch)




