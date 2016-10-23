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

import argparse
from six.moves import xrange
from tensorforce.external.openai_gym import OpenAIGymEnvironment
from tensorforce.util.agent_util import create_agent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-a', '--agent', default='DQNAgent')
    parser.add_argument('-e', '--episodes', type=int, default=100, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=100, help="Maximum number of timesteps per episode")
    parser.add_argument('-m', '--monitor', help="Save results to this file")

    args = parser.parse_args()

    gym_id = args.gym_id

    episodes = args.episodes
    report_episodes = episodes // 10

    max_timesteps = args.max_timesteps

    env = OpenAIGymEnvironment(gym_id)
    agent = create_agent(args.agent, agent_config={}, value_config={}) # TODO: Provide configurations

    if args.monitor:
        env.gym.monitor.start(args.monitor)

    print("Starting {agent_type} for OpenAI Gym '{gym_id}'".format(agent_type=args.agent, gym_id=gym_id))
    for i in xrange(episodes):
        state = env.reset()
        for j in xrange(max_timesteps):
            action = agent.get_action(state)

            result = env.execute_action(action)

            agent.add_observation(state, action, result['reward'], result['terminal_state'])

            state = result['state']
            if result['terminal_state']:
                break

        if i + 1 % report_episodes == 0:
            print("Finished episode {i} after {j} timesteps".format(i=i, j=j))

    if args.monitor:
        env.gym.monitor.close()

    print("DQN learning finished.")
    # TODO: Print results.

if __name__ == '__main__':
    main()