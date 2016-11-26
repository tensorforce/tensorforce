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
from six.moves import xrange

from tensorforce.config import Config
from tensorforce.external.openai_gym import OpenAIGymEnvironment
from tensorforce.util.agent_util import create_agent, get_default_config
from tensorforce.util.wrapper_util import create_wrapper


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment", default='Pong-v0')
    parser.add_argument('-a', '--agent', default='DQNAgent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file",
                        default='examples/configs/dqn_agent.json')
    parser.add_argument('-n', '--network-config', help="Network configuration file",
                        default='examples/configs/dqn_network.json')
    parser.add_argument('-e', '--episodes', type=int, default=1000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=100, help="Maximum number of timesteps per episode")
    parser.add_argument('-m', '--monitor', help="Save results to this file")

    args = parser.parse_args()

    gym_id = args.gym_id

    episodes = args.episodes
    report_episodes = episodes // 10

    max_timesteps = args.max_timesteps

    env = OpenAIGymEnvironment(gym_id)

    # config = Config({
    #     'actions': env.gym.action_space.shape[0],
    #     'action_shape': (env.gym.action_space.shape[0],),
    #     'state_shape': env.gym.observation_space.shape
    # })
    config = Config({
        'actions': env.gym.action_space.n,
        'action_shape': (env.gym.action_space.n,),
        'state_shape': env.gym.observation_space.shape
    })

    if args.agent_config:
        config.read_json(args.agent_config)

    if args.network_config:
        config.read_json(args.network_config)

    agent = create_agent(args.agent, config)

    if args.monitor:
        env.gym.monitor.start(args.monitor)

    state_wrapper = None
    if config.state_wrapper:
        state_wrapper = create_wrapper(config.state_wrapper, config.state_wrapper_param)

    print("Starting {agent_type} for OpenAI Gym '{gym_id}'".format(agent_type=args.agent, gym_id=gym_id))
    i = -1
    for i in xrange(episodes):
        state = env.reset()
        j = -1
        for j in xrange(max_timesteps):
            if state_wrapper:
                full_state = state_wrapper.get_full_state(state)
            else:
                full_state = state
            action = agent.get_action(full_state, i)
            result = env.execute_action(action)

            agent.add_observation(state, action, result['reward'], result['terminal_state'])

            state = result['state']
            if result['terminal_state']:
                break

        if (i + 1) % report_episodes == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=i + 1, ts=j + 1))

    if args.monitor:
        env.gym.monitor.close()

    print("Learning finished. Total episodes: {ep}".format(ep=i + 1))
    # TODO: Print results.


if __name__ == '__main__':
    main()
