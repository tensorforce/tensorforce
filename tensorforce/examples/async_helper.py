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
    parser.add_argument('-p', '--is_ps', default=0, help="Is param server")

    args = parser.parse_args()

    #TODO create cmds, pass through config



if __name__ == '__main__':
    main()
