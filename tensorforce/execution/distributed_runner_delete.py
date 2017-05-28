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
Coordinator for running distributed tensorflow. Starts multiple worker processes, which
themselves use execution classes. Currently to be used in conjunction with the thread runner for
OpenAI universe. More generic distributed API coming.
"""

from copy import deepcopy
import tensorflow as tf

import logging

from tensorforce.agents.distributed_agent import DistributedAgent
from tensorforce.execution.thread_runner import ThreadRunner


class DistributedRunner(object):

    def __init__(self, agent_type, agent_config, network_config, n_agents, n_param_servers, environment,
                 global_steps, max_episode_steps, preprocessor=None, repeat_actions=1,
                 local_steps=20, task_index=0, is_ps=False):

        self.is_ps = is_ps
        self.task_index = task_index
        self.agent_type = agent_type
        self.agent_config = agent_config
        self.network_config = network_config
        self.n_agents = n_agents
        self.n_param_servers = n_param_servers
        self.environment = environment

        self.global_steps = global_steps
        self.max_episode_steps = max_episode_steps
        self.local_steps = local_steps

        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions

        port = 12222

        ps_hosts = []
        worker_hosts = []

        for _ in range(self.n_param_servers):
            ps_hosts.append('127.0.0.1:{}'.format(port))
            port += 1

        for _ in range(self.n_agents):
            worker_hosts.append('127.0.0.1:{}'.format(port))
            port += 1

        cluster = {'ps': ps_hosts, 'worker': worker_hosts}

        self.cluster_spec = tf.train.ClusterSpec(cluster)

        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def run(self):
        """
        Process execution loop.
        """
        # Redirect process output
        # sys.stdout = open('tf_worker_' + str(self.task_index) + '.txt', 'w', 0)
        cluster = self.cluster_spec.as_cluster_def()

        if self.is_ps:
            server = tf.train.Server(cluster, job_name='ps', task_index=self.task_index,
                                     config=tf.ConfigProto(device_filters=["/job:ps"]))

            # Param server does nothing actively
            server.join()
        else:
            # Worker creates runner for execution
            scope = 'worker_' + str(self.task_index)
            self.logger.debug('Creating server')
            server = tf.train.Server(cluster, job_name='worker', task_index=self.task_index,
                                     config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                                           inter_op_parallelism_threads=2,
                                                           log_device_placement=True))
            self.logger.debug('Created server')
            worker_agent = DistributedAgent(self.agent_config, self.network_config, scope, self.task_index, cluster)
            self.logger.debug('Created agent')

            variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
            init_op = tf.variables_initializer(variables_to_save)
            local_init_op = tf.variables_initializer(
                tf.local_variables() + [v for v in tf.global_variables() if v.name.startswith("local")])
            init_all_op = tf.global_variables_initializer()

            def init_fn(sess):
                sess.run(init_all_op)

            config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(self.task_index)])

            supervisor = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                             logdir="/tmp/train_logs",
                                             global_step=worker_agent.model.global_step,
                                             init_op=init_op,
                                             local_init_op=local_init_op,
                                             init_fn=init_fn,
                                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                                             saver=worker_agent.model.saver)
                                             # summary_op=tf.summary.merge_all(),
                                             # summary_writer=worker_agent.model.summary_writer)

            runner = ThreadRunner(worker_agent, deepcopy(self.environment),
                                  self.max_episode_steps, self.local_steps, preprocessor=self.preprocessor,
                                  repeat_actions=self.repeat_actions)

            # Connecting to parameter server
            self.logger.debug('Connecting to session..')
            self.logger.info('Server target = ' + str(server.target))

            with supervisor.managed_session(server.target, config=config) as session, session.as_default():
                self.logger.info('Established session, starting runner..')
                session.run(worker_agent.model.assign_global_to_local)

                runner.start_thread(session)
                self.logger.debug("Runner started")
                global_step_count = worker_agent.get_global_step()
                self.logger.debug("Got global step count")

                while not supervisor.should_stop() and global_step_count < self.global_steps:
                    runner.update()
                    global_step_count = worker_agent.get_global_step()
                    self.logger.debug("Global step count: {}".format(global_step_count))

            self.logger.info('Stopping supervisor')
            supervisor.stop()
