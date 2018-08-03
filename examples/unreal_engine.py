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
Test an Unreal Engine Game as RL-Environment
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import sys
import time
import random
from PIL import Image

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.unreal_engine import UE4Environment


# Users need to give the port on which the UE4 Game listens on for incoming RL-client connections.
# To learn about setting up UE4 Games as RL-environments, go to: https://github.com/ducandu/engine2learn
# - you will need to install the UE4 Engine and the engine2learn plugin
# - supports headless execution of UE4 games under Linux

# python examples/unreal_engine.py -P 6025 -a examples/configs/vpg.json
#   -n examples/configs/mlp2_network.json -e 50000 -m 2000


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-P', '--port', default=6025,
                        help="Port on which the UE4 Game listens on for incoming RL-client connections")
    parser.add_argument('-H', '--host', default=None, help="Hostname of the UE4 Game (default: localhost)")
    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None,
                        help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False,
                        help="Choose actions deterministically")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")
    parser.add_argument('-R', '--random-test-run', action="store_true", help="Do a quick random test run on the env")

    args = parser.parse_args()

    # logging.basicConfig(filename="logfile.txt", level=logging.INFO)
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # We have to connect this remote env to get the specs.
    # We also discretize axis-mappings b/c we will use a deep q-network.
    # Use num_ticks==6 to match Nature paper by Mnih et al.
    # ("human cannot press fire button with more than 10Hz", dt=1/60)
    # TODO: Need to build in capturing and concat'ing last 4 images (plus 8-bit conversion!) into 1 input state signal.
    # TODO: Use pre-processor for that.
    environment = UE4Environment(host=args.host, port=args.port, connect=True, discretize_actions=True, num_ticks=6)
    environment.seed(200)

    # Do a quick random test-run with image capture of the first n images -> then exit after 1000 steps.
    if args.random_test_run:
        # Reset the env.
        s = environment.reset()
        img_format = "RGB" if len(environment.states["shape"]) == 3 else "L"
        img = Image.fromarray(s, img_format)
        # Save first received image as a sanity-check.
        img.save("reset.png")
        for i in range(1000):
            s, is_terminal, r = environment.execute(action=random.choice(range(environment.actions["num_actions"])))
            if i < 10:
                img = Image.fromarray(s, img_format)
                img.save("{:03d}.png".format(i))
            logging.debug("i={} r={} term={}".format(i, r, is_terminal))
            if is_terminal:
                environment.reset()
        quit()

    if args.agent_config is not None:
        with open(args.agent_config, 'r') as fp:
            agent_config = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network_spec is not None:
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

    if args.debug:  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 100

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r, id_):
        if r.episode % report_episodes == 0:
            steps_per_second = r.global_timestep / (time.time() - r.start_time)
            logger.info("Finished episode {} after {} timesteps. SPS={}".format(
                r.global_episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) /
                                                                 min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) /
                                                                 min(100, len(r.episode_rewards))))
        return True

    runner.run(
        timesteps=args.timesteps,
        episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
