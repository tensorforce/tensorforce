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
import numpy as np

from tensorforce import Configuration
from tensorforce.agents import agents as AgentsDictionary
from tensorforce.core.networks import from_json
from tensorforce.execution import ThreadedRunner
from tensorforce.contrib.ale import ALE


"""
To replicate the Asynchronous Methods for Deep Reinforcement Learning paper (https://arxiv.org/abs/1602.01783)
Nstep DQN:
    python threaded_ale.py <path_to_rom> -a DQNNstepAgent -c ./configs/dqn_nstep_agent_visual.json \
    -n ./configs/dqn_2013_network_visual.json -fs 4 -rc -1 1 -ea -w 16

    Note: batch_size in the config should be set to n+1 where n is the desired number of steps
"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('rom', help="File path of the rom")
    parser.add_argument('-a', '--agent', help='Agent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-config', help="Network configuration file")
    parser.add_argument('-w', '--workers', help="Number of threads to run where the model is shared", type=int, default=16)
    parser.add_argument('-fs', '--frame-skip', help="Number of frames to repeat action", type=int, default=1)
    parser.add_argument('-rap', '--repeat-action-probability', help="Repeat action probability", type=float, default=0.0)
    parser.add_argument('-lolt', '--loss-of-life-termination', help="Loss of life counts as terminal state", action='store_true')
    parser.add_argument('-lolr', '--loss-of-life-reward', help="Loss of life reward/penalty. EX: -1 to penalize", type=float, default=0.0)
    parser.add_argument('-ea', '--epsilon-annealing', help='Create separate epislon annealing schedules per thread', action='store_true')
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

    environments = [ALE(args.rom, frame_skip=args.frame_skip,
                        repeat_action_probability=args.repeat_action_probability,
                        loss_of_life_termination=args.loss_of_life_termination,
                        loss_of_life_reward=args.loss_of_life_reward,
                        display_screen=args.display_screen) for t in range(args.workers)]

    if args.network_config:
        network = from_json(args.network_config)
    else:
        network = None
        logger.info("No network configuration provided.")

    agent_configs = []
    for i in range(args.workers):
        agent_config = Configuration.from_json(args.agent_config)

        # optionally overwrite epsilon final values
        if "exploration" in agent_config and "epsilon" in agent_config.exploration.type:
            # epsilon annealing is based on the global step so divide by the total workers
            epsilon_timesteps = agent_config.exploration.epsilon_timesteps // args.workers
            agent_config.exploration.epsilon_timesteps = epsilon_timesteps
            if args.epsilon_annealing:
                # epsilon final values are [0.5, 0.1, 0.01] with probabilities [0.3, 0.4, 0.3]
                epsilon_final = np.random.choice([0.5, 0.1, 0.01], p=[0.3, 0.4, 0.3])
                agent_config.exploration.epsilon_final = epsilon_final

        agent_config.default(dict(states=environments[0].states, actions=environments[0].actions, network=network))
        agent_configs.append(agent_config)

    # let the first agent create the model
    agent = AgentsDictionary[args.agent](config=agent_configs[-1])
    # then create agents with a shared model
    agents = [AgentsDictionary[args.agent](config=agent_configs[t], model=agent.model) for t in range(args.workers - 1)] + [agent]

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

    def episode_finished(stats):
        if args.debug:
            logger.info("Thread {t}. Finished episode {ep} after {ts} timesteps. Reward {r}".
                        format(t=stats['thread_id'], ep=stats['episode'], ts=stats['timestep'], r=stats['episode_reward']))
        return True

    def summary_report(r):
        et = time.time()
        logger.info('=' * 40)
        logger.info('Current Step/Episode: {}/{}'.format(r.global_step, r.global_episode))
        logger.info('SPS: {}'.format(r.global_step / (et - r.start_time)))
        reward_list = r.episode_rewards
        if len(reward_list) > 0:
            logger.info('Max Reward: {}'.format(np.max(reward_list)))
            logger.info("Average of last 500 rewards: {}".format(sum(reward_list[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(reward_list[-100:]) / 100))
        logger.info('=' * 40)

    # create runners
    threaded_runner = ThreadedRunner(agents, environments, repeat_actions=1,
                                     save_path=args.save, save_episodes=args.save_episodes)

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environments[0]))
    threaded_runner.run(summary_interval=100, episode_finished=episode_finished, summary_report=summary_report)
    logger.info("Learning finished. Total episodes: {ep}".format(ep=threaded_runner.global_episode))

    [environments[t].close() for t in range(args.workers)]


if __name__ == '__main__':
    main()
