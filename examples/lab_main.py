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
Deepmind lab execution
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import time
import json

import numpy as np

from tensorforce import TensorForceError
from tensorforce.agents import Agent

# This was necessary for bazel, test if can be removed
logger = logging.getLogger(__name__)

from tensorforce.contrib.deepmind_lab import DeepMindLab
from tensorforce.execution import Runner


def main():
    parser = argparse.ArgumentParser()

    # N.b. if ran from within lab, the working directory is something like lab/bazel-out/../../tensorforce
    # Hence, relative paths will not work without first fetching the path of this run file
    parser.add_argument('-id', '--level-id', default='tests/demo_map',help="DeepMind Lab level id")
    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=1000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=200, help="Maximum number of timesteps per episode")
    parser.add_argument('-m', '--monitor', help="Save results to this directory")
    parser.add_argument('-ms', '--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('-mv', '--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-D', '--debug', action='store_true', default=True, help="Show debug outputs")

    # Redirect output to file
    sys.stdout = open('lab_output.txt', 'w')

    args = parser.parse_args()

    environment = DeepMindLab(args.level_id)

    path = os.path.dirname(__file__)
    if args.agent_config:
        # Use absolute path
        agent_config = json.load(path + args.agent_config)
    else:
        raise TensorForceError("No agent configuration provided.")

    if not args.network_spec:
        raise TensorForceError("No network configuration provided.")
    else:
        network_spec = json.load(path + args.network_config)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # configurable!!!

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
        agent.restore_model(args.load)

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent_config)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )
    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

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

    report_episodes = args.episodes // 1000

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            sps = r.total_timesteps / (time.time() - r.start_time)
            logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode, ts=r.timestep, sps=sps))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(np.mean(r.episode_rewards[-500:])))
            logger.info("Average of last 100 rewards: {}".format(np.mean(r.episode_rewards[-100:])))
        return True

    logger.info("Starting {agent} for Lab environment '{env}'".format(agent=agent, env=environment))
    runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)
    runner.close()
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode + 1))

    environment.close()


if __name__ == '__main__':
    main()
