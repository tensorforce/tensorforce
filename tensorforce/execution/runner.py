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
Simple runner for non-realtime single process execution, appropriate for
OpenAI gym.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

from six.moves import xrange
import tensorflow as tf

from tensorforce import TensorForceError


class Runner(object):

    # These agents can be used in an A3C fashion.
    async_supported = ('VPGAgent', 'PPOAgent')  # And potentially TRPOAgent, needs to be checked...

    def __init__(self, agent, environment, repeat_actions=1, cluster_spec=None, task_index=None, save_path=None, save_episodes=None):
        """
        Initialize a Runner object.

        Args:
            agent: `Agent` object containing the reinforcement learning agent
            environment: `../../environments/Environment` object containing
            repeat_actions:
            cluster_spec:
            task_index:
            save_path:
            save_episodes:
        """
        if cluster_spec is not None and str(agent) not in Runner.async_supported:
            raise TensorForceError('Agent type not supported for distributed runner.')
        self.agent = agent
        self.environment = environment
        self.repeat_actions = repeat_actions
        self.cluster_spec = cluster_spec
        self.task_index = task_index
        self.save_path = save_path
        self.save_episodes = save_episodes

    def run(self, episodes=-1, max_timesteps=-1, deterministic=False, episode_finished=None):
        """
        Runs an environments for the specified number of episodes and time steps per episode.

        Args:
            episodes: Number of episodes to execute
            max_timesteps: Max timesteps in a given episode
            episode_finished: Optional termination condition, e.g. a particular mean reward threshold

        Returns:

        """
        if self.cluster_spec is not None:
            assert self.task_index is not None
            # Redirect process output
            # sys.stdout = open('tf_worker_' + str(self.task_index) + '.txt', 'w', 0)
            cluster_def = self.cluster_spec.as_cluster_def()

            if self.task_index == -1:
                server = tf.train.Server(
                    server_or_cluster_def=cluster_def,
                    job_name='ps',
                    task_index=0,
                    config=tf.ConfigProto(device_filters=["/job:ps"])
                )
                # Param server does nothing actively
                server.join()
                return

            # Worker creates runner for execution.
            server = tf.train.Server(
                server_or_cluster_def=cluster_def,
                job_name='worker',
                task_index=self.task_index,
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=2,
                    log_device_placement=True
                )
            )

            config = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:{}/cpu:0'.format(self.task_index)])

            init_op = tf.global_variables_initializer()
            supervisor = tf.train.Supervisor(
                is_chief=(self.task_index == 0),
                logdir='/tmp/train_logs',
                global_step=self.agent.model.global_timestep,
                init_op=tf.variables_initializer(self.agent.model.global_variables),
                local_init_op=tf.variables_initializer(self.agent.model.variables),
                init_fn=(lambda session: session.run(init_op)),
                saver=self.agent.model.saver)
            # summary_op=tf.summary.merge_all(),
            # summary_writer=worker_agent.model.summary_writer)

            managed_session = supervisor.managed_session(server.target, config=config)
            session = managed_session.__enter__()
            self.agent.model.set_session(session)
            # session.run(self.agent.model.update_local)

        # Keep track of episode reward and episode length for statistics.
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []

        self.total_timesteps = 0
        self.episode = 1
        self.start_time = time.time()
        while True:
            state = self.environment.reset()
            self.agent.reset()
            episode_reward = 0

            self.timestep = 0
            episode_start_time = time.time()
            while True:
                action = self.agent.act(state=state, deterministic=deterministic)
                if self.repeat_actions > 1:
                    reward = 0
                    for repeat in xrange(self.repeat_actions):
                        state, step_reward, terminal = self.environment.execute(action=action)
                        reward += step_reward
                        if terminal:
                            break
                else:
                    state, reward, terminal = self.environment.execute(action=action)

                self.agent.observe(reward=reward, terminal=terminal)

                self.timestep += 1
                self.total_timesteps += 1
                episode_reward += reward

                if terminal or self.timestep == max_timesteps:
                    break

            self.agent.observe_episode_reward(episode_reward)
            time_passed = time.time() - episode_start_time
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.timestep)
            self.episode_times.append(time_passed)

            if self.save_path and self.save_episodes is not None and self.episode % self.save_episodes == 0:
                print("Saving agent after episode {}".format(self.episode))
                self.agent.save_model(self.save_path)

            if episode_finished and not episode_finished(self):
                return
            if self.cluster_spec is None:
                if self.episode >= episodes:
                    return
            elif session.run(self.agent.model.global_episode) >= episodes:
                return
            self.episode += 1

        if self.cluster_spec is not None:
            managed_session.__exit__(None, None, None)
            supervisor.stop()
