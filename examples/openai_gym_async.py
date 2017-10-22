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
$ python examples/openai_gym_async.py Pong-ram-v0 -a VPGAgent -c examples/configs/vpg_agent.json -n examples/configs/vpg_network.json -w 3 -D

You can check what the workers are doing:
$ tmux a -t openai_async  # `ctrl+b d` to exit tmux

To kill the session:
$ python examples/openai_gym_async.py Pong-ram-v0 -a VPGAgent -c examples/configs/vpg_agent.json -n examples/configs/vpg_network.json -w 3 -D -K
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inspect
import logging
import os
import sys
import time

import tensorflow as tf
from six.moves import xrange, shlex_quote

from tensorforce import Configuration, TensorForceError
from tensorforce.agents import agents
from tensorforce.core.networks import from_json
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.util import log_levels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-a', '--agent', help='Agent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-config', help="Network configuration file")
    parser.add_argument('-e', '--episodes', type=int, default=50000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-w', '--num-workers', type=int, default=1, help="Number of worker agents")
    parser.add_argument('-m', '--monitor', help="Save results to this file")
    parser.add_argument('-M', '--mode', choices=['tmux', 'child'], default='tmux', help="Starter mode")
    parser.add_argument('-L', '--logdir', default='logs_async', help="Log directory")
    parser.add_argument('-C', '--is-child', action='store_true')
    parser.add_argument('-i', '--task-index', type=int, default=0, help="Task index")
    parser.add_argument('-K', '--kill', action='store_true', default=False, help="Kill runners")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    session_name = 'openai_async'
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

        def build_cmd(index):
            cmd_args = [
                'CUDA_VISIBLE_DEVICES=',
                sys.executable, target_script,
                args.gym_id,
                '--is-child',
                '--agent', args.agent,
                '--agent-config', os.path.join(os.getcwd(), args.agent_config),
                '--network-config', os.path.join(os.getcwd(), args.network_config),
                '--num-workers', args.num_workers,
                '--task-index', index
            ]
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
        cmds.append(wrap_cmd(session_name, 'ps', build_cmd(-1)))

        for i in xrange(args.num_workers):
            name = 'w_{}'.format(i)
            if args.mode == 'tmux':
                cmds.append('tmux new-window -t {} -n {} -d {}'.format(session_name, name, shell))
            cmds.append(wrap_cmd(session_name, name, build_cmd(i)))

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

    if args.agent_config:
        agent_config = Configuration.from_json(args.agent_config)
    else:
        raise TensorForceError("No agent configuration provided.")
    if not args.network_config:
        raise TensorForceError("No network configuration provided.")
    agent_config.default(dict(states=environment.states, actions=environment.actions, network=from_json(args.network_config)))

    agent_config.default(dict(distributed=True, cluster_spec=cluster_spec, global_model=(args.task_index == -1), device=('/job:ps' if args.task_index == -1 else '/job:worker/task:{}/cpu:0'.format(args.task_index))))

    logger = logging.getLogger(__name__)
    logger.setLevel(log_levels[agent_config.log_level])

    # don't write summaries for tasks other than chief (0)
    if args.task_index != 0:
        agent_config.tf_summary = None
        agent_config.tf_summary_level = -1
        logger.info("Summaries disabled for agent {}".format(args.task_index))
    agent = agents[args.agent](config=agent_config)

    logger.info("Starting distributed agent for OpenAI Gym '{gym_id}'".format(gym_id=args.gym_id))
    logger.info("Config:")
    logger.info(agent_config)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1,
        cluster_spec=cluster_spec,
        task_index=args.task_index,
        save_path=args.logdir
    )

    report_episodes = args.episodes // 1000
    if args.debug:
        report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            sps = r.total_timesteps / (time.time() - r.start_time)
            logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode, ts=r.timestep, sps=sps))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)


if __name__ == '__main__':
    main()
