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

from six.moves import xrange
import tensorflow as tf

from tensorforce import TensorForceError


class Runner(object):

    # These agents can be used in an A3C fashion
    async_supported = ['DQNAgent', 'VPGAgent', 'NAFAgent']

    def __init__(self, agent, environment, repeat_actions=1, preprocessor=None, cluster_spec=None, task_index=None, save_path=None, save_episodes=None):
        if cluster_spec is not None and str(agent) not in Runner.async_supported:
            raise TensorForceError('Agent type not supported for distributed runner.')
        self.agent = agent
        self.environment = environment
        self.repeat_actions = repeat_actions
        self.preprocessor = preprocessor
        self.cluster_spec = cluster_spec
        self.task_index = task_index
        self.save_path = save_path
        self.save_episodes = save_episodes

    def run(self, episodes=-1, max_timesteps=-1, episode_finished=None, before_execution=None):
        """
        Runs an environments for the specified number of episodes and time steps per episode.
        
        Args:
            episodes: Number of episodes to execute
            max_timesteps: Max timesteps in a given episode
            episode_finished: Optional termination condition, e.g. a particular mean reward threshold
            before_execution: Optional filter function to apply to action before execution

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

            # Worker creates runner for execution
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

            variables_to_save = [v for v in tf.global_variables() if not v.name.startswith('local')]
            init_op = tf.variables_initializer(variables_to_save)
            local_init_op = tf.variables_initializer(tf.local_variables() + [v for v in tf.global_variables() if v.name.startswith('local')])
            init_all_op = tf.global_variables_initializer()

            def init_fn(sess):
                sess.run(init_all_op)

            config = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:{}/cpu:0'.format(self.task_index)])

            supervisor = tf.train.Supervisor(
                is_chief=(self.task_index == 0),
                logdir='/tmp/train_logs',
                global_step=self.agent.model.global_step,
                init_op=init_op,
                local_init_op=local_init_op,
                init_fn=init_fn,
                ready_op=tf.report_uninitialized_variables(variables_to_save),
                saver=self.agent.model.saver)
            # summary_op=tf.summary.merge_all(),
            # summary_writer=worker_agent.model.summary_writer)

            # # Connecting to parameter server
            # self.logger.debug('Connecting to session..')
            # self.logger.info('Server target = ' + str(server.target))

            # with supervisor.managed_session(server.target, config=config) as session, session.as_default():
            # self.logger.info('Established session, starting runner..')
            managed_session = supervisor.managed_session(server.target, config=config)
            session = managed_session.__enter__()
            self.agent.model.session = session
            # session.run(self.agent.model.update_local)

        # save episode reward and length for statistics
        self.episode_rewards = []
        self.episode_lengths = []

        self.episode = 1
        while True:
            state = self.environment.reset()
            self.agent.reset()
            episode_reward = 0

            self.timestep = 1
            while True:
                if self.preprocessor:
                    processed_state = self.preprocessor.process(state)
                else:
                    processed_state = state

                action = self.agent.act(state=processed_state)

                if before_execution:
                    action = before_execution(self, action)

                if self.repeat_actions > 1:
                    reward = 0
                    for repeat in xrange(self.repeat_actions):
                        state, step_reward, terminal = self.environment.execute(action=action)
                        reward += step_reward
                        if terminal:
                            break
                else:
                    state, reward, terminal = self.environment.execute(action=action)

                episode_reward += reward
                self.agent.observe(state=processed_state, action=action, reward=reward, terminal=terminal)

                if terminal or self.timestep == max_timesteps:
                    break
                self.timestep += 1

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.timestep)

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
