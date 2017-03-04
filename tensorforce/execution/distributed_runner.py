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
themselves use execution classes. Runners can be threaded for realtime usage (e.g. OpenAI universe)
"""

from copy import deepcopy
from multiprocessing import Process
import time
import tensorflow as tf
import sys
import os

from tensorforce.agents.distributed_agent import DistributedAgent
from tensorforce.execution.thread_runner import ThreadRunner


class DistributedRunner(object):
    def __init__(self, agent_type, agent_config, n_agents, n_param_servers, environment,
                 max_global_steps=1000000, max_episode_steps=1000, local_steps=20, preprocessor=None, repeat_actions=1):

        self.max_episode_steps = max_episode_steps
        self.agent_type = agent_type
        self.agent_config = agent_config
        self.n_agents = n_agents
        self.n_param_servers = n_param_servers
        self.environment = environment

        # Overall steps as monitored by the distributed runner
        self.max_global_steps = max_global_steps

        # Max steps in a given episode until we break
        self.max_episode_steps = max_episode_steps

        # Max local steps by a worker before we update the policy
        self.local_steps = local_steps

        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions

        # This follows the OpenAI start agent logic only that we use the multiprocessing module
        # instead of cmd line calls for distributed tensorflow which is slightly more convenient
        # to debug

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

    def run(self):
        """
        Creates and starts worker processes and parameter servers.
        """

        # TODO we don't use this currently? -> remove
        self.processes = []

        for index in range(self.n_param_servers):
            process = Process(target=process_worker,
                              args=(self, index, self.max_global_steps, self.max_episode_steps,
                                    self.local_steps, False))
            self.processes.append(process)

            process.start()

        for index in range(self.n_agents):
            process = Process(target=process_worker,
                              args=(self, index, self.max_global_steps, self.max_episode_steps,
                                    self.local_steps, False))
            self.processes.append(process)

            process.start()


def process_worker(master, index, max_steps,
                   max_episode_steps, local_steps, is_param_server=False):
    """
    Process execution loop.

    :param master:
    :param index:
    :param episodes:
    :param max_episode_steps:
    :param is_param_server:

    """

    # Redirect process output
    sys.stdout = open('tf_worker_' + str(index) + '.txt', 'w')
    cluster = master.cluster_spec.as_cluster_def()

    if is_param_server:
        server = tf.train.Server(cluster, job_name='ps', task_index=index,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))

        # Param server does nothing actively
        server.join()
    else:
        # Worker creates runner for execution
        scope = 'worker_' + str(index)

        server = tf.train.Server(cluster, job_name='worker', task_index=index,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                                       inter_op_parallelism_threads=2,
                                                       log_device_placement=True))

        worker_agent = DistributedAgent(master.agent_config, scope, index, cluster)

        def init_fn(sess):
            sess.run(worker_agent.model.init_op)

        config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(index)])

        supervisor = tf.train.Supervisor(is_chief=(index == 0),
                                         logdir="/tmp/train_logs",
                                         global_step=worker_agent.model.global_step,
                                         init_op=tf.global_variables_initializer(),
                                         init_fn=init_fn,
                                         saver=worker_agent.model.saver,
                                         summary_op=tf.summary.merge_all(),
                                         summary_writer=worker_agent.model.summary_writer)

        global_steps = max_steps

        runner = ThreadRunner(worker_agent, deepcopy(master.environment),
                              max_episode_steps, local_steps, preprocessor=master.preprocessor,
                              repeat_actions=master.repeat_actions)

        # Connecting to parameter server
        print('Connecting to session..')
        print('Server target = ' + str(server.target))

        with supervisor.managed_session(server.target, config=config) as session, session.as_default():
            print('Established session, starting runner..')

            runner.start_thread(session)
            global_step_count = worker_agent.increment_global_step()

            while not supervisor.should_stop() and global_step_count < global_steps:
                runner.update()
                global_step_count = worker_agent.increment_global_step()

        print('Stopping supervisor')
        supervisor.stop()


        # def get_episode_finished_handler(self, condition):
        #     def episode_finished(execution):
        #         condition.acquire()
        #         condition.wait()
        #         return self.continue_execution
        #     return episode_finished

        # def run(self, episodes, max_episode_steps, episode_finished=None):
        #     self.total_states = 0
        #     self.episode_rewards = []
        #     self.continue_execution = True

        #     runners = []
        #     processes = []
        #     conditions = []
        #     for agent, environment in zip(self.agents, self.environments):
        #         condition = Condition()
        #         conditions.append(condition)

        #         execution = Runner(agent, environment, preprocessor=self.preprocessor, repeat_actions=self.repeat_actions)  # deepcopy?
        #         runners.append(execution)

        #         thread = Thread(target=execution.run, args=(episodes, max_episode_steps), kwargs={'episode_finished': self.get_episode_finished_handler(condition)})
        #         processes.append(thread)
        #         thread.start()

        #     self.episode = 0
        #     loop = True
        #     while loop:
        #         for condition, execution in zip(conditions, runners):
        #             if condition._waiters:
        #                 self.timestep = execution.timestep
        #                 self.episode += 1
        #                 self.episode_rewards.append(execution.episode_rewards[-1])
        #                 # perform async update of parameters
        #                 # if T mod Itarget == 0:
        #                 #     update target network
        #                 # clear gradient
        #                 # sync parameters
        #                 condition.acquire()
        #                 condition.notify()
        #                 condition.release()
        #                 if self.episode >= episodes or (episode_finished and not episode_finished(self)):
        #                     loop = False
        #                     break
        #         self.total_states = sum(execution.total_states for execution in runners)

        #     self.continue_execution = False
        #     stopped = 0
        #     while stopped < self.n_runners:
        #         for condition, thread in zip(conditions, processes):
        #             if condition._waiters:
        #                 condition.acquire()
        #                 condition.notify()
        #                 condition.release()
        #                 conditions.remove(condition)
        #             if not thread.is_alive():
        #                 processes.remove(thread)
        #                 stopped += 1
