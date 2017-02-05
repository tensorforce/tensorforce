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

from copy import deepcopy
from threading import Condition, Thread

import tensorflow as tf

from tensorforce.runner import Runner
from tensorforce.util.agent_util import create_agent


class AsyncRunner(Runner):
    def __init__(self, agent_type, agent_config, n_agents, environment, preprocessor=None, repeat_actions=1):
        super(AsyncRunner, self).__init__(
            create_agent(agent_type, agent_config + {'tf_device': 'replica', 'tf_worker_device': '/job:master'}),
            environment, preprocessor=preprocessor, repeat_actions=repeat_actions)
        self.agent_type = agent_type
        self.agent_config = agent_config
        self.n_agents = n_agents
        ps_hosts = ['127.0.0.1:12222']
        worker_hosts = ['127.0.0.1:{}'.format(n) for n in range(12223, 12223 + n_agents)]
        self.cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    def run(self, episodes, max_timesteps, episode_finished=None):
        self.threads = []
        self.continue_execution = True

        # global_episode = tf.get_variable('global_episode', shape=(), dtype=tf.int32, initializer=tf.zeros_initializer, trainable=False)
        # server = tf.train.Server(self.cluster_spec.as_cluster_def(), job_name='ps', task_index=0)

        # supervisor = tf.train.Supervisor(
        #     is_chief=master,
        #     logdir="/tmp/train_logs",
        #     init_op=init_op,
        #     summary_op=summary_op,
        #     saver=saver,
        #     global_step=global_episode,
        #     save_model_secs=600)

        # with supervisor.managed_session(server.target) as session:

        for index in range(self.n_agents):
            thread = Thread(target=worker_thread, args=(self, index, episodes, max_timesteps))
            self.threads.append(thread)
            thread.start()

            #     while not supervisor.should_stop() and global_episode < episodes:
            #         global_episode = session.run(global_episode)
            # supervisor.stop()

            # self.continue_execution = False
            # while self.threads:
            #     for thread in list(self.threads):
            #         if not thread.is_alive():
            #             self.threads.remove(thread)


def worker_thread(master, index, episodes, max_timesteps):
    if not master.continue_execution:
        return

    worker_device = '/job:worker{}'.format(index)

    with tf.device(worker_device):
        global_episode = tf.get_variable('global_episode', shape=(), dtype=tf.int32, initializer=tf.zeros_initializer,
                                         trainable=False)
    server = tf.train.Server(master.cluster_spec.as_cluster_def(), job_name='worker', task_index=index)

    worker_agent = create_agent(master.agent_type, master.agent_config + {'tf_device': worker_device})
    ps_agent = create_agent(master.agent_type,
                            master.agent_config + {'tf_device': 'replica', 'tf_worker_device': worker_device})

    supervisor = tf.train.Supervisor(
        is_chief=(index == 0),
        # logdir="/tmp/train_logs",
        init_op=worker_agent.model.init_op,
        # summary_op=summary_op,
        saver=worker_agent.model.saver,
        global_step=global_episode,
        summary_writer=worker_agent.model.writer)

    worker = Runner(worker_agent, deepcopy(master.environment), preprocessor=master.preprocessor,
                    repeat_actions=master.repeat_actions)

    with supervisor.managed_session(server.target) as session:
        def episode_finished(r):
            print('Episode finished')
            grads, _ = zip(*r.agent.get_gradients())
            grads_and_vars = list(zip(grads, ps_agent.get_variables()))
            ps_agent.apply_gradients(grads_and_vars)
            r.assign_variables(ps_agent.get_variables())
            increment_episode_op = global_episode.assign_add(1)
            session.run(increment_episode_op)  # necessary?

            return master.continue_execution

        worker.run(episodes, max_timesteps, episode_finished=episode_finished)

    supervisor.stop()












    # def get_episode_finished_handler(self, condition):
    #     def episode_finished(runner):
    #         condition.acquire()
    #         condition.wait()
    #         return self.continue_execution
    #     return episode_finished

    # def run(self, episodes, max_timesteps, episode_finished=None):
    #     self.total_states = 0
    #     self.episode_rewards = []
    #     self.continue_execution = True

    #     runners = []
    #     threads = []
    #     conditions = []
    #     for agent, environment in zip(self.agents, self.environments):
    #         condition = Condition()
    #         conditions.append(condition)

    #         runner = Runner(agent, environment, preprocessor=self.preprocessor, repeat_actions=self.repeat_actions)  # deepcopy?
    #         runners.append(runner)

    #         thread = Thread(target=runner.run, args=(episodes, max_timesteps), kwargs={'episode_finished': self.get_episode_finished_handler(condition)})
    #         threads.append(thread)
    #         thread.start()

    #     self.episode = 0
    #     loop = True
    #     while loop:
    #         for condition, runner in zip(conditions, runners):
    #             if condition._waiters:
    #                 self.timestep = runner.timestep
    #                 self.episode += 1
    #                 self.episode_rewards.append(runner.episode_rewards[-1])
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
    #         self.total_states = sum(runner.total_states for runner in runners)

    #     self.continue_execution = False
    #     stopped = 0
    #     while stopped < self.n_runners:
    #         for condition, thread in zip(conditions, threads):
    #             if condition._waiters:
    #                 condition.acquire()
    #                 condition.notify()
    #                 condition.release()
    #                 conditions.remove(condition)
    #             if not thread.is_alive():
    #                 threads.remove(thread)
    #                 stopped += 1
