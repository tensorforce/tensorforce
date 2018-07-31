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
Asynchronous OpenAI gym execution - this currently implements A3C semantics
but execution semantics will be changed to have more explicit control.

To run this script with 3 workers on Pong-ram-v0:
$ python examples/openai_gym_async.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000 -W 3

Or on CartPole-v0:
$ python examples/openai_gym_async.py CartPole-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 10000 -m 200 -W 3

You can check what the workers are doing:
$ tmux a -t OpenAI  # `ctrl+b d` to exit tmux

To kill the session:
$ python examples/openai_gym_async.py CartPole-v0 -W 3 -K
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

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', help="Choose actions deterministically")
    parser.add_argument('-M', '--mode', choices=('tmux', 'child'), default='tmux', help="Starter mode")
    parser.add_argument('-W', '--num-workers', type=int, default=1, help="Number of worker agents")
    parser.add_argument('-C', '--child', action='store_true', help="Child process")
    parser.add_argument('-P', '--parameter-server', action='store_true', help="Parameter server")
    parser.add_argument('-I', '--task-index', type=int, default=0, help="Task index")
    parser.add_argument('-K', '--kill', action='store_true', help="Kill runners")
    parser.add_argument('-L', '--logdir', default='logs_async', help="Log directory")
    parser.add_argument('-D', '--debug', action='store_true', help="Show debug outputs")

    args = parser.parse_args()

    session_name = 'OpenAI-' + args.gym_id
    shell = '/bin/bash'

    kill_cmds = [
        "kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1".format(12222 + args.num_workers),
        "tmux kill-session -t {}".format(session_name),
    ]
    if args.kill:
        os.system("\n".join(kill_cmds))
        return 0

    if not args.child:
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
                # 'CUDA_VISIBLE_DEVICES=',
                sys.executable, target_script, args.gym_id,
                '--agent', os.path.join(os.getcwd(), args.agent),
                '--network', os.path.join(os.getcwd(), args.network),
                '--num-workers', args.num_workers,
                '--child',
                '--task-index', index
            ]
            if args.episodes is not None:
                cmd_args.append('--episodes')
                cmd_args.append(args.episodes)
            if args.timesteps is not None:
                cmd_args.append('--timesteps')
                cmd_args.append(args.timesteps)
            if args.max_episode_timesteps is not None:
                cmd_args.append('--max-episode-timesteps')
                cmd_args.append(args.max_episode_timesteps)
            if args.deterministic:
                cmd_args.append('--deterministic')
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

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # log_levels[agent.log_level])

    stdout_logger = logging.StreamHandler(sys.stdout)
    stdout_logger.setLevel(logging.INFO)
    logger.addHandler(stdout_logger)

    if args.agent is not None:
        with open(args.agent, 'r') as fp:
            agent = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network is not None:
        with open(args.network, 'r') as fp:
            network = json.load(fp=fp)
    else:
        network = None
        logger.info("No network configuration provided.")

    if args.parameter_server:
        agent['device'] = '/job:ps/task:{}'.format(args.task_index)  # '/cpu:0'
    else:
        agent['device'] = '/job:worker/task:{}'.format(args.task_index)  # '/cpu:0'

    agent['execution'] = dict(
        type='distributed',
        distributed_spec=dict(
            cluster_spec=cluster_spec,
            task_index=args.task_index,
            job='ps' if args.parameter_server else 'worker',
            protocol='grpc'
        )
    )

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network
        )
    )

    logger.info("Starting distributed agent for OpenAI Gym '{gym_id}'".format(gym_id=args.gym_id))
    logger.info("Config:")
    logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if args.debug:  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 100

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {} after overall {} timesteps. Steps Per Second {}".format(
                r.agent.episode,
                r.agent.timestep,
                steps_per_second)
            )
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        return True

    runner.run(
        timesteps=args.timesteps,
        episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished
    )
    runner.close()


if __name__ == '__main__':
    main()
