# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================
"""
Coordinator for running distributed tensorflow. Starts multiple worker processes, which
themselves use execution classes. Runners can be threaded for realtime usage (e.g. OpenAI universe)
"""

from copy import deepcopy
from multiprocessing import Process
import time
import tensorflow as tf

from tensorforce.agents.distributed_agent import DistributedAgent
from tensorforce.execution.thread_runner import ThreadRunner
from tensorforce.util.agent_util import create_agent


class DistributedRunner(object):
    def __init__(self, agent_type, agent_config, n_agents, n_param_servers, environment,
                 episodes, max_timesteps, preprocessor=None, repeat_actions=1):

        self.agent_type = agent_type
        self.agent_config = agent_config
        self.n_agents = n_agents
        self.n_param_servers = n_param_servers
        self.environment = environment
        self.episodes = episodes
        self.max_timesteps = max_timesteps

        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions

        port = 12222
        ps_hosts = ['127.0.0.1:{}'.format(n) for n in range(port, port + self.n_param_servers)]
        worker_hosts = ['127.0.0.1:{}'.format(n) for n in range(self.n_param_servers,
                                                                self.n_param_servers + n_agents)]

        self.cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    def run(self):
        """
        Creates and starts worker processes and parameter servers.
        """
        self.processes = []

        for index in range(self.n_param_servers):
            process = Process(target=process_worker, args=(self, index, self.episodes, self.max_timesteps, True))
            self.processes.append(process)
            process.start()

        for index in range(self.n_agents):
            process = Process(target=process_worker, args=(self, index, self.episodes, self.max_timesteps, False))
            self.processes.append(process)

            process.start()


def process_worker(master, index, episodes, max_timesteps, is_param_server=False):
    """
    Process execution loop.

    :param master:
    :param index:
    :param episodes:
    :param max_timesteps:
    :param is_param_server:

    """
    if not master.continue_execution:
        return

    worker_device = '/job:worker{}'.format(index)

    with tf.device(worker_device):
        global_step_count = tf.get_variable('global_step_count', shape=(), dtype=tf.int32, initializer=tf.zeros_initializer,
                                         trainable=False)

    if is_param_server:
        server = tf.train.Server(master.cluster_spec.as_cluster_def(), job_name='ps', task_index=index,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))

        # Param server does nothing actively
        while True:
            time.sleep(1000)
    else:
        # Worker creates runner for execution
        scope = 'worker_' + str(index)
        server = tf.train.Server(master.cluster_spec.as_cluster_def(), job_name='worker', task_index=index)

        worker_agent = DistributedAgent(master.agent_config + {'tf_device': worker_device}, scope)

        supervisor = tf.train.Supervisor(
            is_chief=(index == 0),
            logdir="/tmp/train_logs",
            init_op=worker_agent.model.init_op,
            # summary_op=summary_op,
            saver=worker_agent.model.saver,
            global_step=global_step_count,
            summary_writer=worker_agent.model.writer)

        global_steps = 10000000

        runner = ThreadRunner(worker_agent, deepcopy(master.environment),
                              episodes, 20, preprocessor=master.preprocessor,
                              repeat_actions=master.repeat_actions)

        config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(index)])

        # Connecting to parameter server
        with supervisor.managed_session(server.target, config) as session:
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

        # def run(self, episodes, max_timesteps, episode_finished=None):
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

        #         thread = Thread(target=execution.run, args=(episodes, max_timesteps), kwargs={'episode_finished': self.get_episode_finished_handler(condition)})
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
