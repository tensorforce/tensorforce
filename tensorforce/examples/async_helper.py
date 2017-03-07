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
Creates the cmd line call for the distributed starter, which is necessary due to TensorFlow not supporting multiprocessing.
"""
from six.moves import xrange

import sys
import inspect
import argparse
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
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
    parser.add_argument('-i', '--task_index', default=0, help="Task index")

    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    target_script = os.path.join(this_dir, 'openai_gym_async.py')
    def build_cmd(index, parameter_server):
        cmd_args = [sys.executable, target_script,
                args.gym_id,
                '--agent-config', os.path.join(os.getcwd(), args.agent_config),
                '--network-config', os.path.join(os.getcwd(), args.network_config),
                '--num-workers', args.num_workers,
                '--task_index', index,
                '--is_ps', parameter_server
                ]
        print(cmd_args)
        return cmd_args


    #TODO create one call for worker, parameter server, also kill old grpcs
    cmds = []
    for i in xrange(args.num_workers):
        cmds.append(build_cmd(i, 0))
        # cmds += "python openai_gym_async.py Pong-ram-v0 -c examples/configs/vpg_agent.json -n examples/configs/vpg_network.json" \
        #         " -w " + str(args.num_workers) + " -i" + str(i) + " -p 0 && "

    # add one PS call
    cmds.append(build_cmd(0, 1))
    # cmds += "python openai_gym_async.py Pong-ram-v0 -c examples/configs/vpg_agent.json -n examples/configs/vpg_network.json" \
    #         " -w " + str(args.num_workers) + " -i 0 -p 1"

    print(cmds)

    # os.system("\n".join(cmds))


if __name__ == '__main__':
    main()
