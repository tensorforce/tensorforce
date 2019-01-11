# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

import argparse
import importlib
import os
import time

from tensorforce.agents import Agent
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.execution import Runner


# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000

# python examples/openai_gym.py CartPole-v1 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 2000 -m 200


def main():
    parser = argparse.ArgumentParser()
    # Gym arguments
    parser.add_argument('-g', '--gym', help="Gym environment id")
    parser.add_argument(
        '-i', '--import-modules', help="Import module(s) required for gym environment"
    )
    parser.add_argument('--monitor', type=str, default=None, help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")
    # Agent arguments
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    # Runner arguments
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument(
        '-m', '--max-episode-timesteps', type=int, default=None,
        help="Maximum number of timesteps per episode"
    )
    parser.add_argument(
        '-d', '--deterministic', action='store_true', default=False,
        help="Choose actions deterministically"
    )
    args = parser.parse_args()

    if args.import_modules is not None:
        for module in args.import_modules.split(','):
            importlib.import_module(name=module)

    environment = OpenAIGym(
        gym_id=args.gym, monitor=args.monitor, monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video, visualize=args.visualize
    )

    agent = Agent.from_spec(
        spec=args.agent, states=environment.states(), actions=environment.actions(),
        network=args.network
    )

    runner = Runner(agent=agent, environment=environment)

    def callback(r):
        if r.episode % 100 == 0:
            print(
                "================================================\n"
                "Average secs/episode over 100 episodes: {time:0.2f}\n"
                "Average steps/sec over 100 episodes:    {timestep:0.2f}\n"
                "Average reward over 100 episodes:       {reward100:0.2f}\n"
                "Average reward over 500 episodes:       {reward500:0.2f}".format(
                    time=(sum(r.episode_times[-100:]) / 100.0),
                    timestep=(sum(r.episode_timesteps[-100:]) / sum(r.episode_times[-100:])),
                    reward100=(sum(r.episode_rewards[-100:]) / min(100.0, r.episode)),
                    reward500=(sum(r.episode_rewards[-500:]) / min(500.0, r.episode))
                )
            )
        return True

    runner.run(
        num_timesteps=args.timesteps, num_episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps, deterministic=args.deterministic,
        callback=callback
    )

    runner.close()


if __name__ == '__main__':
    main()
