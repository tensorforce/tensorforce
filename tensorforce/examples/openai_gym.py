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
OpenAI gym execution
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse

import numpy as np

from tensorforce.config import Config
from tensorforce.external.openai_gym import OpenAIGymEnvironment
from tensorforce.util.experiment_util import build_preprocessing_stack
from tensorforce.util.agent_util import create_agent
from tensorforce.execution import Runner


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-a', '--agent', default='DQNAgent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-config', help="Network configuration file")
    parser.add_argument('-e', '--episodes', type=int, default=50000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=200, help="Maximum number of timesteps per episode")
    parser.add_argument('-m', '--monitor', help="Save results to this directory")
    parser.add_argument('-ms', '--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('-mv', '--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    gym_id = args.gym_id

    env = OpenAIGymEnvironment(gym_id, monitor=args.monitor, monitor_safe=args.monitor_safe, monitor_video=args.monitor_video)

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

    if args.debug:
        print("-" * 16)
        print("File configuration:")
        print(config)

    agent = create_agent(args.agent, config)

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.load_model(args.load)

    if args.debug:
        print("-" * 16)
        print("Agent configuration:")
        print(config)

    runner = Runner(agent, env, preprocessor=stack, repeat_actions=config.repeat_actions)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))
        runner.save_model(args.save, args.save_episodes)

    report_episodes = args.episodes // 10
    if args.debug:
        report_episodes = 10

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
    env.close()


if __name__ == '__main__':
    main()
