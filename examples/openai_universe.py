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
OpenAI universe example.

In order to use openai universe, please make sure you have docker installed.

Then use like this:

``python examples/openai_universe.py -a DQNAgent -c examples/configs/dqn.json -n examples/configs/cnn_dqn_network.json flashgames.DuskDrive-v0``

This will create a docker session that you can connect to by visiting

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import time

from tensorforce import TensorForceError
from tensorforce.agents import Agent
import json
from tensorforce.execution import Runner
from tensorforce.contrib.openai_universe import OpenAIUniverse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=50000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=2000*60, help="Maximum number of timesteps per episode")
    # parser.add_argument('-m', '--monitor', help="Save results to this directory")
    # parser.add_argument('-ms', '--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    # parser.add_argument('-mv', '--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    environment = OpenAIUniverse(args.gym_id)
    environment.configure(remotes=1)

    if args.agent_config is not None:
        with open(args.agent_config, 'r') as fp:
            agent_config = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network_spec:
        with open(args.network_spec, 'r') as fp:
            network_spec = json.load(fp=fp)
    else:
        network_spec = None
        logger.info("No network configuration provided.")

    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network_spec
        )
    )

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.load_model(args.load)

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent_config)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    report_episodes = args.episodes // 1000
    if args.debug:
        report_episodes = 1
    def episode_finished(r):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {} after {} timesteps. Steps Per Second {}".format(
                r.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
    runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)
    runner.close()
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

    # if args.monitor:
    #     environment.gym.monitor.close()
    environment.close()


if __name__ == '__main__':
    main()
