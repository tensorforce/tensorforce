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
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import inspect
import argparse

from six.moves import xrange, shlex_quote

from tensorforce.config import Config, create_config
from tensorforce.execution.distributed_runner import DistributedRunner
from tensorforce.external.openai_gym import OpenAIGymEnvironment
from tensorforce.util.experiment_util import build_preprocessing_stack


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
    # Currently does not do anything since we don't have the distributed API for all models yet
    parser.add_argument('-a', '--agent', default='DQNAgent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file",
                        default='examples/configs/dqn_agent.json')
    parser.add_argument('-n', '--network-config', help="Network configuration file",
                        default='examples/configs/dqn_network.json')
    parser.add_argument('-e', '--global-steps', type=int, default=1000000, help="Total number of steps")
    parser.add_argument('-t', '--max-timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-l', '--local-steps', type=int, default=20, help="Maximum number of local steps before update")
    parser.add_argument('-w', '--num-workers', type=int, default=1, help="Number of worker agents")
    parser.add_argument('-r', '--repeat-actions', type=int, default=1, help="???")
    parser.add_argument('-m', '--monitor', help="Save results to this file")
    parser.add_argument('-M', '--mode', choices=['tmux', 'child'], default='tmux', help="Starter mode")
    parser.add_argument('-L', '--logdir', default='logs_async', help="Log directory")
    parser.add_argument('-C', '--is-child', action='store_true', default=False)
    parser.add_argument('-i', '--task-index', type=int, default=0, help="Task index")
    parser.add_argument('-p', '--is-ps', type=int, default=0, help="Is param server")
    parser.add_argument('-K', '--kill', action='store_true', default=False, help="Kill runners")

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

        def build_cmd(index, parameter_server):
            cmd_args = ['CUDA_VISIBLE_DEVICES=',
                        sys.executable, target_script,
                        args.gym_id,
                        '--is-child',
                        '--agent-config', os.path.join(os.getcwd(), args.agent_config),
                        '--network-config', os.path.join(os.getcwd(), args.network_config),
                        '--num-workers', args.num_workers,
                        '--task-index', index,
                        '--is-ps', parameter_server
                        ]

            return cmd_args

        if args.mode == 'tmux':
            cmds = kill_cmds + ['tmux new-session -d -s {} -n ps'.format(session_name)]
        elif args.mode == 'child':
            cmds = ['mkdir -p {}'.format(args.logdir),
                    'rm -f {}/kill.sh'.format(args.logdir),
                    'echo "#/bin/bash" > {}/kill.sh'.format(args.logdir),
                    'chmod +x {}/kill.sh'.format(args.logdir)]
        cmds.append(wrap_cmd(session_name, 'ps', build_cmd(0, 1)))

        for i in xrange(args.num_workers):
            name = 'w_{}'.format(i)
            if args.mode == 'tmux':
                cmds.append('tmux new-window -t {} -n {} -d {}'.format(session_name, name, shell))
            cmds.append(wrap_cmd(session_name, name, build_cmd(i, 0)))


        # add one PS call
        # cmds.append('tmux new-window -t {} -n ps -d {}'.format(session_name, shell))

        print("\n".join(cmds))

        os.system("\n".join(cmds))

        return 0


    env = OpenAIGymEnvironment(args.gym_id)

    config = Config({
        'repeat_actions': 1,
        'actions': env.actions,
        'action_shape': env.action_shape,
        'state_shape': env.state_shape
    })

    if args.agent_config:
        config.read_json(args.agent_config)
    if args.network_config:
        config.read_json(args.network_config)

    preprocessing_config = config.get('preprocessing')

    if preprocessing_config:
        stack = build_preprocessing_stack(preprocessing_config)
        config.state_shape = stack.shape(config.state_shape)
    else:
        stack = None

    print("Starting distributed agent for OpenAI Gym '{gym_id}'".format(gym_id=args.gym_id))
    print("Config:")
    print(config)

    runner = DistributedRunner(agent_type=args.agent, agent_config=config, n_agents=args.num_workers, n_param_servers=1,
                               environment=env, global_steps=args.global_steps, max_episode_steps=args.max_timesteps,
                               preprocessor=stack, repeat_actions=args.repeat_actions, local_steps=args.local_steps,
                               task_index=args.task_index, is_ps=(args.is_ps == 1))
    runner.run()


if __name__ == '__main__':
    main()
