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

"""
OpenAI gym runner
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse

import numpy as np

from tensorforce.config import Config
from tensorforce.external.openai_gym import OpenAIGymEnvironment
from tensorforce.util.agent_util import create_agent, get_default_config
from tensorforce.runner import Runner

from tensorforce import preprocessing


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-a', '--agent', default='DQNAgent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file",
                        default='examples/configs/dqn_agent.json')
    parser.add_argument('-n', '--network-config', help="Network configuration file",
                        default='examples/configs/dqn_network.json')
    parser.add_argument('-e', '--episodes', type=int, default=10000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=200, help="Maximum number of timesteps per episode")
    parser.add_argument('-m', '--monitor', help="Save results to this file")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")


    args = parser.parse_args()

    gym_id = args.gym_id

    env = OpenAIGymEnvironment(gym_id)

    config = Config({
        'actions': env.actions,
        'action_shape': env.action_shape,
        'state_shape': env.state_shape
    })

    if args.agent_config:
        config.read_json(args.agent_config)

    if args.network_config:
        config.read_json(args.network_config)

    # TODO: make stack configurable
    #stack = preprocessing.Stack()
    #stack += preprocessing.Maximum(2)
    #stack += preprocessing.Grayscale()
    #stack += preprocessing.Imresize([84, 84])
    #stack += preprocessing.Concat(4)

    #config.state_shape = stack.shape(config.state_shape)

    if args.debug:
        print("-" * 16)
        print("File configuration:")
        print(config)

    agent = create_agent(args.agent, config)

    if args.debug:
        print("-" * 16)
        print("Agent configuration:")
        print(config)

    runner = Runner(agent, env, preprocessor=None, repeat_actions=1)

    if args.monitor:
        env.gym.monitor.start(args.monitor)
        env.gym.monitor.configure(video_callable=lambda count: False)  # count % 500 == 0)

    report_episodes = args.episodes // 10
    if args.debug:
        report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 500 rewards: {}".format(np.mean(r.episode_rewards[-500:])))
            print("Average of last 100 rewards: {}".format(np.mean(r.episode_rewards[-100:])))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))
    runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)
    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode + 1))

    if args.monitor:
        env.gym.monitor.close()


if __name__ == '__main__':
    main()
