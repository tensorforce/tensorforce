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
OpenAI universe execution
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
from six.moves import xrange

import numpy as np

from tensorforce.config import Config
from tensorforce.external.openai_universe import OpenAIUniverseEnvironment
from tensorforce.util.agent_util import create_agent, get_default_config
from tensorforce.util.wrapper_util import create_wrapper


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('env_id', help="ID of the universe environment", default='HarvestDay-v0')
    parser.add_argument('-a', '--agent', default='DQNAgent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file",
                        default='examples/configs/dqn_agent.json')
    parser.add_argument('-n', '--network-config', help="Network configuration file",
                        default='examples/configs/dqn_network.json')
    parser.add_argument('-e', '--episodes', type=int, default=10000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-m', '--monitor', help="Save results to this file")


    args = parser.parse_args()

    env_id = args.env_id

    episodes = args.episodes
    report_episodes = episodes / 100

    max_timesteps = args.max_timesteps

    environment = OpenAIUniverseEnvironment(env_id)

    config = Config({
        'actions': environment.actions,
        'action_shape': environment.action_shape,
        'state_shape': environment.state_shape
    })

    if args.agent_config:
        config.read_json(args.agent_config)

    if args.network_config:
        config.read_json(args.network_config)

    state_wrapper = None
    if config.state_wrapper:
        state_wrapper = create_wrapper(config.state_wrapper, config.state_wrapper_param)
        config.state_shape = state_wrapper.state_shape(config.state_shape)

    agent = create_agent(args.agent, config)

    if args.monitor:
        environment.env.monitor.start(args.monitor)
        environment.env.monitor.configure(video_callable=lambda count: False) # count % 500 == 0)

    print("Starting {agent_type} for OpenAI Universe environment '{env_id}'".format(agent_type=args.agent, env_id=env_id))
    total_states = 0
    repeat_actions = config.get('repeat_actions', 4)
    episode_rewards = []
    for i in xrange(episodes):
        state = environment.reset()
        episode_reward = 0
        repeat_action_count = 0
        for j in xrange(max_timesteps):
            if state_wrapper:
                full_state = state_wrapper.get_full_state(state)
            else:
                full_state = state
            if repeat_action_count <= 0:
                action = agent.get_action(full_state, i, total_states)
                repeat_action_count = repeat_actions - 1
            else:
                repeat_action_count -= 1
            result = environment.execute_action(action)
            episode_reward += result['reward']
            agent.add_observation(full_state, action, result['reward'], result['terminal_state'])

            state = result['state']
            total_states += 1
            if result['terminal_state']:
                break

        episode_rewards.append(episode_reward)

        if i % report_episodes == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=i + 1, ts=j + 1))
            print("Total reward: {}".format(episode_reward))
            print("Average of last 500 rewards: {}".format(np.mean(episode_rewards[-500:])))
            print("Average of last 100 rewards: {}".format(np.mean(episode_rewards[-100:])))

    if args.monitor:
        environment.env.monitor.close()

    print("Learning finished. Total episodes: {ep}".format(ep=i + 1))
    # TODO: Print results.


if __name__ == '__main__':
    main()
