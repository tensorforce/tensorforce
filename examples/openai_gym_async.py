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
OpenAI gym execution

To run this script with 3 workers:
$ python examples/openai_gym_async.py Pong-ram-v0 -a vpg_agent -c examples/configs/vpg_agent.json -n examples/configs/vpg_network.json -w 3 -D

You can check what the workers are doing:
$ tmux a -t openai_async  # `ctrl+b d` to exit tmux

To kill the session:
$ python examples/openai_gym_async.py Pong-ram-v0 -a vpg_agent -c examples/configs/vpg_agent.json -n examples/configs/vpg_network.json -w 3 -D -K
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inspect
import json
import logging
import os
import sys
import time

import tensorflow as tf
from six.moves import xrange, shlex_quote

from tensorforce import Configuration
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-a', '--agent', help='Agent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=50000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-w', '--num-workers', type=int, default=1, help="Number of worker agents")
    parser.add_argument('-m', '--monitor', help="Save results to this file")
    parser.add_argument('-M', '--mode', choices=['tmux', 'child'], default='tmux', help="Starter mode")
    parser.add_argument('-L', '--logdir', default='logs_async', help="Log directory")
    parser.add_argument('-K', '--kill', action='store_true', default=False, help="Kill runners")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")
    parser.add_argument('-C', '--is-child', action='store_true')
    parser.add_argument('-P', '--parameter-server', action='store_true', help="Parameter server")
    parser.add_argument('-i', '--task-index', type=int, default=0, help="Task index")

    args = parser.parse_args()

    session_name = 'OpenAI'
    shell = '/bin/bash'

    kill_cmds = [
        "kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1".format(12222 + args.num_workers),
        "tmux kill-session -t {}".format(session_name),
    ]
    if args.kill:
        os.system("\n".join(kill_cmds))
        return 0

    if not args.is_child:
        # start up child processes
        target_script = os.path.abspath(inspect.stack()[0][1])

        def wrap_cmd(session, name, cmd):
            if isinstance(cmd, list):
                cmd = ' '.join(shlex_quote(str(arg)) for arg in cmd)
            if args.mode == 'tmux':
                return 'tmux send-keys -t {}:{} {} Enter'.format(session, name, shlex_quote(cmd))
            elif args.mode == 'child':
                return '{} > {}/{}.{}.out 2>&1 & echo kill $! >> {}/kill.sh'.format(
                    cmd, args.logdir, session, name, args.logdir
                )

        def build_cmd(ps, index):
            cmd_args = [
                'CUDA_VISIBLE_DEVICES=',
                sys.executable, target_script,
                args.gym_id,
                '--is-child',
                '--agent', args.agent,
                '--agent-config', os.path.join(os.getcwd(), args.agent_config),
                '--network-spec', os.path.join(os.getcwd(), args.network_spec),
                '--num-workers', args.num_workers,
                '--task-index', index
            ]
            if ps:
                cmd_args.append('--parameter-server')
            if args.debug:
                cmd_args.append('--debug')
            return cmd_args

        if args.mode == 'tmux':
            cmds = kill_cmds + ['tmux new-session -d -s {} -n ps'.format(session_name)]
        elif args.mode == 'child':
            cmds = ['mkdir -p {}'.format(args.logdir),
                    'rm -f {}/kill.sh'.format(args.logdir),
                    'echo "#/bin/bash" > {}/kill.sh'.format(args.logdir),
                    'chmod +x {}/kill.sh'.format(args.logdir)]

        cmds.append(wrap_cmd(session_name, 'ps', build_cmd(ps=True, index=0)))

        for i in xrange(args.num_workers):
            name = 'worker{}'.format(i)
            if args.mode == 'tmux':
                cmds.append('tmux new-window -t {} -n {} -d {}'.format(session_name, name, shell))
            cmds.append(wrap_cmd(session_name, name, build_cmd(ps=False, index=i)))

        # add one PS call
        # cmds.append('tmux new-window -t {} -n ps -d {}'.format(session_name, shell))

        print("\n".join(cmds))

        os.system("\n".join(cmds))

        return 0

    ps_hosts = ['127.0.0.1:{}'.format(12222)]
    worker_hosts = []
    port = 12223
    for _ in range(args.num_workers):
        worker_hosts.append('127.0.0.1:{}'.format(port))
        port += 1
    cluster = {'ps': ps_hosts, 'worker': worker_hosts}
    cluster_spec = tf.train.ClusterSpec(cluster)

    environment = OpenAIGym(args.gym_id)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # log_levels[agent_config.log_level])

    if args.agent_config:
        config = Configuration.from_json(args.agent_config)
    else:
        config = Configuration()
        logger.info("No agent configuration provided.")

    config.obligatory(
        cluster_spec=cluster_spec,
        parameter_server=args.parameter_server,
        task_index=args.task_index,
        device=('/job:{}/task:{}'.format('ps' if args.parameter_server else 'worker', args.task_index)),  # '/cpu:0'
        local_model=(not args.parameter_server)
    )

    if args.network_spec:
        with open(args.network_spec, 'r') as fp:
            network_spec = json.load(fp=fp)
    else:
        network_spec = None
        logger.info("No network configuration provided.")

    agent = Agent.from_spec(
        spec=args.agent,
        kwargs=dict(
            states_spec=environment.states,
            actions_spec=environment.actions,
            network_spec=network_spec,
            config=config
        )
    )

    logger.info("Starting distributed agent for OpenAI Gym '{gym_id}'".format(gym_id=args.gym_id))
    logger.info("Config:")
    logger.info(config)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    report_episodes = args.episodes // 1000
    if args.debug:
        report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {} after overall {} timesteps. Steps Per Second {}".format(r.agent.episode, r.agent.timestep, steps_per_second))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)


if __name__ == '__main__':
    main()
