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
Maze Explorer execution
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import time

from tensorforce import Configuration
from tensorforce.agents import agents
from tensorforce.core.networks import from_json
from tensorforce.execution import Runner
from tensorforce.contrib.maze_explorer import MazeExplorer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help="ID of the game mode")
    parser.add_argument('--hide', dest='hide', action='store_const', const=True, default=False, help="Hide output window")
    parser.add_argument('-a', '--agent', help='Agent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-config', help="Network configuration file")
    parser.add_argument('-e', '--episodes', type=int, default=50000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # configurable!!!

    environment = MazeExplorer(mode_id=args.mode, visible=not args.hide)

    if args.agent_config:
        agent_config = Configuration.from_json(args.agent_config)
    else:
        agent_config = Configuration()
        logger.info("No agent configuration provided.")
    if args.network_config:
        network = from_json(args.network_config)
    else:
        network = None
        logger.info("No network configuration provided.")
    agent_config.default(dict(states=environment.states, actions=environment.actions, network=network))
    agent = agents[args.agent](config=agent_config)

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.load_model(args.load)

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent_config)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1,
        save_path=args.save,
        save_episodes=args.save_episodes
    )

    report_episodes = args.episodes // 1000
    if args.debug:
        report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            sps = r.total_timesteps / (time.time() - r.start_time)
            logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode, ts=r.timestep, sps=sps))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
    runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

    if args.monitor:
        environment.gym.monitor.close()
    environment.close()


if __name__ == '__main__':
    main()
