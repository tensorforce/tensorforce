"""
Deepmind Pycolab usage example

For setting up Pycolab see https://github.com/deepmind/pycolab
For Pycolab game definition example check: examples/extraterrestrial_maurauders.py
Each game definition should have a make_game() function that returns Pycolab Game Engine object
and a get_ui() function that returns Pycolab UI object.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import json
import logging
import os
import time
import sys

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.dm_pycolab import DMPycolab

# python examples/dm_pycolab.py examples/extraterrestrial_maurauders.py -a examples/configs/dqn.json -n examples/configs/mlp2_embedding_flatten_network.json -e 5000 -m 300

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('path',  help="Path to Pycolab game definition file")
    parser.add_argument('-i', '--import-modules', help="Import module(s) required for environment")
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable Pycolab game's visualization")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if args.import_modules is not None:
        for module in args.import_modules.split(','):
            importlib.import_module(name=module)

    if args.path is not None:
        sys.path.append(os.path.dirname(os.path.expanduser(args.path)))
        game_name = os.path.splitext(os.path.basename(args.path))[0]

        try:
	    game_env = importlib.import_module(game_name)
        except:
            raise TensorForceError("Could not get game {0} from path {1}".format(game_name, args.path))

    environment = DMPycolab(
        game=game_env.make_game(),
        ui=game_env.get_ui(),
        visualize=args.visualize
    )

    if args.agent is not None:
        with open(args.agent, 'r') as fp:
            agent = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network is not None:
        with open(args.network, 'r') as fp:
            network = json.load(fp=fp)
    else:
        network = None
        logger.info("No network configuration provided.")

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network,
        )
    )

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

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
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        if args.save and args.save_episodes is not None and not r.episode % args.save_episodes:
            logger.info("Saving agent to {}".format(args.save))
            r.agent.save_model(args.save)

        return True

    runner.run(
        num_timesteps=args.timesteps,
        num_episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        episode_finished=episode_finished,
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))

if __name__ == '__main__':
    main()
