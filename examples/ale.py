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
Arcade Learning Environment execution
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import time

from tensorforce import TensorForceError
import json

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.ale import ALE


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('rom', help="File path of the rom")
    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")
    parser.add_argument('-fs', '--frame-skip', help="Number of frames to repeat action", type=int, default=1)
    parser.add_argument('-rap', '--repeat-action-probability', help="Repeat action probability", type=float, default=0.0)
    parser.add_argument('-lolt', '--loss-of-life-termination', help="Loss of life counts as terminal state", action='store_true')
    parser.add_argument('-lolr', '--loss-of-life-reward', help="Loss of life reward/penalty. EX: -1 to penalize", type=float, default=0.0)
    parser.add_argument('-ds', '--display-screen', action='store_true', default=False, help="Display emulator screen")
    parser.add_argument('-e', '--episodes', type=int, default=50000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # configurable!!!
    logger.addHandler(logging.StreamHandler(sys.stdout))

    environment = ALE(args.rom, frame_skip=args.frame_skip,
                      repeat_action_probability=args.repeat_action_probability,
                      loss_of_life_termination=args.loss_of_life_termination,
                      loss_of_life_reward=args.loss_of_life_reward,
                      display_screen=args.display_screen)

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
        repeat_actions=1
    )

    report_episodes = args.episodes // 1000
    if args.debug:
        report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            sps = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode, ts=r.timestep, sps=sps))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
    runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)
    runner.close()
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

    environment.close()


if __name__ == '__main__':
    main()
