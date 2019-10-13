# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import importlib
import json
import os

import matplotlib
import numpy as np
import tensorflow as tf

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(v=tf.logging.ERROR)


def main():
    parser = argparse.ArgumentParser(description='Tensorforce runner')
    parser.add_argument(
        'agent', help='Agent (configuration JSON file, name, or library module)'
    )
    parser.add_argument(
        'environment',
        help='Environment (name, configuration JSON file, or library module)'
    )
    # Agent arguments
    parser.add_argument(
        '-n', '--network', type=str, default=None,
        help='Network (configuration JSON file, name, or library module)'
    )
    # Environment arguments
    parser.add_argument(
        '-l', '--level', type=str, default=None,
        help='Level or game id, like `CartPole-v1`, if supported'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize agent--environment interaction, if supported'
    )
    parser.add_argument(
        '--visualize-directory', type=str, default=None,
        help='Directory to store videos of agent--environment interaction, if supported'
    )
    parser.add_argument(
        '-i', '--import-modules', type=str, default=None,
        help='Import comma-separated modules required for environment'
    )
    # Runner arguments
    parser.add_argument('-t', '--timesteps', type=int, default=None, help='Number of timesteps')
    parser.add_argument('-e', '--episodes', type=int, default=None, help='Number of episodes')
    parser.add_argument(
        '-m', '--max-episode-timesteps', type=int, default=None,
        help='Maximum number of timesteps per episode'
    ),
    parser.add_argument(
        '--mean-horizon', type=int, default=10,
        help='Number of timesteps/episodes for mean reward computation'
    )
    parser.add_argument('-v', '--evaluation', action='store_true', help='Evaluation mode')
    parser.add_argument(
        '-s', '--save-best-agent', action='store_true', help='Save best-performing agent'
    )
    # Logging arguments
    parser.add_argument('-r', '--repeat', type=int, default=1, help='Number of repetitions')
    parser.add_argument(
        '-p', '--path', type=str, default=None,
        help='Logging path, directory plus filename without extension'
    )
    parser.add_argument('--seaborn', action='store_true', help='Use seaborn')
    args = parser.parse_args()

    if args.import_modules is not None:
        for module in args.import_modules.split(','):
            importlib.import_module(name=module)

    if args.path is None:
        callback = None

    else:
        assert os.path.splitext(args.path)[1] == ''
        assert args.episodes is not None and args.visualize is not None
        rewards = [list() for _ in range(args.episodes)]
        timesteps = [list() for _ in range(args.episodes)]
        seconds = [list() for _ in range(args.episodes)]
        agent_seconds = [list() for _ in range(args.episodes)]

        def callback(r):
            rewards[r.episodes - 1].append(r.episode_reward)
            timesteps[r.episodes - 1].append(r.episode_timestep)
            seconds[r.episodes - 1].append(r.episode_second)
            agent_seconds[r.episodes - 1].append(r.episode_agent_second)
            return True

    if args.visualize:
        if args.level is None:
            environment = Environment.create(environment=args.environment, visualize=True)
        else:
            environment = Environment.create(
                environment=args.environment, level=args.level, visualize=True
            )

    elif args.visualize_directory:
        if args.level is None:
            environment = Environment.create(
                environment=args.environment, visualize_directory=args.visualize_directory
            )
        else:
            environment = Environment.create(
                environment=args.environment, level=args.level,
                visualize_directory=args.visualize_directory
            )

    else:
        if args.level is None:
            environment = Environment.create(environment=args.environment)
        else:
            environment = Environment.create(environment=args.environment, level=args.level)

    for _ in range(args.repeat):
        agent_kwargs = dict()
        if args.network is not None:
            agent_kwargs['network'] = args.network
        if args.max_episode_timesteps is not None:
            assert environment.max_episode_timesteps() is None or \
                environment.max_episode_timesteps() == args.max_episode_timesteps
            agent_kwargs['max_episode_timesteps'] = args.max_episode_timesteps
        agent = Agent.create(agent=args.agent, environment=environment, **agent_kwargs)

        runner = Runner(agent=agent, environment=environment)
        runner.run(
            num_timesteps=args.timesteps, num_episodes=args.episodes,
            max_episode_timesteps=args.max_episode_timesteps, callback=callback,
            mean_horizon=args.mean_horizon, evaluation=args.evaluation
            # save_best_model=args.save_best_model
        )
        runner.close()

    if args.path is not None:
        directory = os.path.split(args.path)[0]
        if directory != '' and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

        with open(args.path + '.json', 'w') as filehandle:
            filehandle.write(
                json.dumps(dict(
                    rewards=rewards, timesteps=timesteps, seconds=seconds,
                    agent_seconds=agent_seconds
                ))
            )

        if args.seaborn:
            import seaborn as sns
            sns.set()

        xs = np.arange(len(rewards))
        min_rewards = np.amin(rewards, axis=1)
        max_rewards = np.amax(rewards, axis=1)
        median_rewards = np.median(rewards, axis=1)
        plt.plot(xs, median_rewards, color='green', linewidth=2.0)
        plt.fill_between(xs, min_rewards, max_rewards, color='green', alpha=0.4)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.savefig(fname=(args.path + '.png'))


if __name__ == '__main__':
    main()
