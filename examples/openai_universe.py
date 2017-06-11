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

``python examples/openai_universe.py -a DQNAgent -c examples/configs/dqn_agent.json -n examples/configs/dqn_network.json flashgames.DuskDrive-v0``

This will create a docker session that you can connect to by visiting

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import numpy as np

import go_vncdriver

from tensorforce import Configuration, TensorForceError
from tensorforce.agents import create_agent
from tensorforce.core.model import log_levels
from tensorforce.core.preprocessing import build_preprocessing_stack
from tensorforce.environments.openai_universe import OpenAIUniverse
from tensorforce.execution import Runner


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-a', '--agent', default='DQNAgent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-config', help="Network configuration file")
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

    env = OpenAIUniverse(args.gym_id)
    env.configure(remotes=1)

    default = dict(
        repeat_actions=1,
        actions=env.actions,
        states=env.states,
        max_episode_length=args.max_timesteps
    )

    if args.agent_config:
        config = Configuration.from_json(args.agent_config)
    else:
        config = Configuration()

    config.default(default)

    if args.network_config:
        network_config = Configuration.from_json(args.network_config).network_layers
    else:
        if config.network_layers:
            network_config = config.network_layers
        else:
            raise TensorForceError("Error: No network configuration provided.")

    if args.debug:
        print("Configuration:")
        print(config)

    logger = logging.getLogger(__name__)
    logger.setLevel(log_levels[config['loglevel']])

    # preprocessing_config = config['preprocessing']
    # if preprocessing_config:
    #     stack = build_preprocessing_stack(preprocessing_config)
    #     config.states['shape'] = stack.shape(config.states['shape'])
    # else:
    stack = None

    agent = create_agent(args.agent, config, network_config)

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.load_model(args.load)

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(config)

    runner = Runner(agent, env, preprocessor=stack, repeat_actions=config.repeat_actions)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))
        runner.save_model(args.save, args.save_episodes)

    report_episodes = args.episodes // 1000
    if args.debug:
        report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            logger.info("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(np.mean(r.episode_rewards[-500:])))
            logger.info("Average of last 100 rewards: {}".format(np.mean(r.episode_rewards[-100:])))
        return True

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))
    runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode + 1))

    if args.monitor:
        env.gym.monitor.close()
    env.close()


if __name__ == '__main__':
    main()
