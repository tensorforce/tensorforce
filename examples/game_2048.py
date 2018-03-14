"""
Game 2048 execution
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import logging
import json

from tensorforce import TensorForceError
from tensorforce.execution import Runner
from tensorforce.agents import Agent
from tensorforce.contrib.game_2048 import Game2048


# python examples/game_2048.py -a examples/configs/ppo.json -n examples/configs/mlp2_network.json

# python examples/game_2048.py -a examples/configs/ppo_cnn.json -n examples/configs/cnn_network_2048.json


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"))
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")

    args = parser.parse_args()

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

    if network_spec[0]['type'] == 'conv2d':
        agent_config['states_preprocessing'] = [{'type': 'expand_dims',
                                                 'axis': -1}]
    else:
        agent_config['states_preprocessing'] = [{'type': 'flatten'}]

    logger.info("Start training")

    environment = Game2048()

    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network_spec,
        )
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    def episode_finished(r):
        if r.episode % 100 == 0:
            sps = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode,
                                                                                                    ts=r.timestep,
                                                                                                    sps=sps))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Episode timesteps: {}".format(r.episode_timestep))
            logger.info("Episode largest tile: {}".format(r.environment.largest_tile))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    runner.run(
        timesteps=6000000,
        episodes=1000,
        max_episode_timesteps=10000,
        deterministic=False,
        episode_finished=episode_finished
    )

    terminal = False
    state = environment.reset()
    while not terminal:
        action = agent.act(state)
        state, terminal, reward = environment.execute(action)
    environment.print_state()

    runner.close()


if __name__ == '__main__':
    main()
