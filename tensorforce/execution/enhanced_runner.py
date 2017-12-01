
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.execution.runner import Runner
from tqdm import tqdm
from collections import OrderedDict

import time
import numpy as np
from six.moves import xrange


class EnhancedRunner(Runner):

  def writeOut(self, s):
      tqdm.write(s)


  def run(
        self,
        timesteps=None,
        episodes=None,
        max_episode_timesteps=None,
        deterministic=False,
        episode_finished=None
    ):
        """
        Runs the agent on the environment.
        Args:
            timesteps: Number of timesteps
            episodes: Number of episodes
            max_episode_timesteps: Max number of timesteps per episode
            deterministic: Deterministic flag
            episode_finished: Function handler taking a `Runner` argument and returning a boolean indicating
                whether to continue execution. For instance, useful for reporting intermediate performance or
                integrating termination conditions.
        """
        print("working 1")

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()

        self.agent.reset()

        self.episode = self.agent.episode
        if episodes is not None:
            episodes += self.agent.episode

        self.timestep = self.agent.timestep
        if timesteps is not None:
            timesteps += self.agent.timestep

        print("working 2")

        total = 10000000000

        pbar = tqdm(range(episodes))

        for i in pbar:
            #pbar.set_description("Processing episode ", i)
            episode_start_time = time.time()
            total = total + 100

            #print("working 3")

            self.agent.reset()
            state = self.environment.reset()
            episode_reward = 0
            self.episode_timestep = 0

            while True:
                pbar.set_description("Processing episode %s" % i)
                if(i>1):
                    pbar.set_postfix(OrderedDict([
                        ('R', '{:8.0f}'.format(self.episode_rewards[-1])),
                        ('AR100', '{:8.2f}'.format(np.mean(self.episode_rewards[-100:]))),
                        ('AR500', '{:8.2f}'.format(np.mean(self.episode_rewards[-500:])))
                        ]))

                total = total + 1
                action = self.agent.act(states=state, deterministic=deterministic)

                #print("working 4")

                if self.repeat_actions > 1:
                    reward = 0

                    for repeat in trange(self.repeat_actions):
                        #print("working")
                        state, terminal, step_reward = self.environment.execute(actions=action)
                        reward += step_reward
                        if terminal:
                            break
                else:
                    state, terminal, reward = self.environment.execute(actions=action)

                if max_episode_timesteps is not None and self.episode_timestep >= max_episode_timesteps:
                    terminal = True

                self.agent.observe(terminal=terminal, reward=reward)

                self.episode_timestep += 1
                self.timestep += 1
                episode_reward += reward

                if terminal or self.agent.should_stop():  # TODO: should_stop also termina?
                    break

            time_passed = time.time() - episode_start_time

            self.episode_rewards.append(episode_reward)
            self.episode_timesteps.append(self.episode_timestep)
            self.episode_times.append(time_passed)

            self.episode += 1

            if episode_finished and not episode_finished(self) or \
                    (episodes is not None and self.agent.episode >= episodes) or \
                    (timesteps is not None and self.agent.timestep >= timesteps) or \
                    self.agent.should_stop():
                # agent.episode / agent.timestep are globally updated
                break

        self.agent.close()
