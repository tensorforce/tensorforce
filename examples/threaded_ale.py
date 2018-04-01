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
Arcade Learning Environment execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

from six.moves import xrange
import argparse
import logging
import os
import sys
import time
import numpy as np

from tensorforce import TensorForceError
from tensorforce.agents import agents as AgentsDictionary, Agent
import json
from tensorforce.execution import ThreadedRunner
from tensorforce.contrib.ale import ALE
from tensorforce.execution.threaded_runner import WorkerAgentGenerator

"""
To replicate the Asynchronous Methods for Deep Reinforcement Learning paper (https://arxiv.org/abs/1602.01783)
Nstep DQN:
    python threaded_ale.py breakout.bin -a configs/dqn_visual.json -n 
    configs/cnn_dqn2013_network.json -fs 4 -ea -w 4


    Note: batch_size in the config should be set to n+1 where n is the desired number of steps
"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('rom', help="File path of the rom")
    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")
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
                        display_screen=args.display_screen) for _ in range(args.workers)]

    if args.network_spec:
        with open(args.network_spec, 'r') as fp:
            network_spec = json.load(fp=fp)
    else:
        network_spec = None
        logger.info("No network configuration provided.")

    agent_configs = []
    if args.agent_config is not None:
        with open(args.agent_config, 'r') as fp:
            agent_config = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    for i in range(args.workers):
        worker_config = deepcopy(agent_config)

        # Optionally overwrite epsilon final values
        if "explorations_spec" in worker_config and worker_config['explorations_spec']['type'] == "epsilon_anneal":
            if args.epsilon_annealing:
                # epsilon final values are [0.5, 0.1, 0.01] with probabilities [0.3, 0.4, 0.3]
                epsilon_final = np.random.choice([0.5, 0.1, 0.01], p=[0.3, 0.4, 0.3])
                worker_config['explorations_spec']["epsilon_final"] = epsilon_final

        agent_configs.append(worker_config)

    # Let the first agent create the model
    # Manually assign model
    logger.info(agent_configs[0])

    agent = Agent.from_spec(
        spec=agent_configs[0],
        kwargs=dict(
            states=environments[0].states,
            actions=environments[0].actions,
            network=network_spec
        )
    )

    agents = [agent]

    for i in xrange(args.workers - 1):
        config = agent_configs[i]
        agent_type = config.pop('type', None)
        worker = WorkerAgentGenerator(AgentsDictionary[agent_type])(
            states=environments[0].states,
            actions=environments[0].actions,
            network=network_spec,
            model=agent.model,
            **config
        )
        agents.append(worker)

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent_configs[0])

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    def episode_finished(stats):
        if args.debug:
            logger.info(
                "Thread {t}. Finished episode {ep} after {ts} timesteps. Reward {r}".
                format(t=stats['thread_id'], ep=stats['episode'], ts=stats['timestep'], r=stats['episode_reward'])
            )
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

    # Create runners
    threaded_runner = ThreadedRunner(
        agents,
        environments,
        repeat_actions=1,
        save_path=args.save,
        save_episodes=args.save_episodes
    )

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environments[0]))
    threaded_runner.run(summary_interval=100, episode_finished=episode_finished, summary_report=summary_report)
    threaded_runner.close()
    logger.info("Learning finished. Total episodes: {ep}".format(ep=threaded_runner.global_episode))


if __name__ == '__main__':
    main()
