# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from tensorforce.agents import Agent
from tensorforce.core.utils.json_encoder import NumpyJSONEncoder
from tensorforce.environments import Environment
from tensorforce.execution import Runner

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Tensorforce runner')
    # Agent arguments
    parser.add_argument(
        '-a', '--agent', type=str, default=None,
        help='Agent (name, configuration JSON file, or library module)'
    )
    parser.add_argument(
        '-n', '--network', type=str, default=None,
        help='Network (name, configuration JSON file, or library module)'
    )
    # Environment arguments
    parser.add_argument(
        '-e', '--environment', type=str, default=None,
        help='Environment (name, configuration JSON file, or library module)'
    )
    parser.add_argument(
        '-l', '--level', type=str, default=None,
        help='Level or game id, like `CartPole-v1`, if supported'
    )
    parser.add_argument(
        '-m', '--max-episode-timesteps', type=int, default=None,
        help='Maximum number of timesteps per episode'
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
        '--import-modules', type=str, default=None,
        help='Import comma-separated modules required for environment'
    )
    # Parallel execution arguments
    parser.add_argument(
        '--num-parallel', type=int, default=None,
        help='Number of environment instances to execute in parallel'
    )
    parser.add_argument(
        '--batch-agent-calls', action='store_true',
        help='Batch agent calls for parallel environment execution'
    )
    parser.add_argument(
        '--sync-timesteps', action='store_true',
        help='Synchronize parallel environment execution on timestep-level'
    )
    parser.add_argument(
        '--sync-episodes', action='store_true',
        help='Synchronize parallel environment execution on episode-level'
    )
    parser.add_argument(
        '--remote', type=str, choices=('multiprocessing', 'socket-client', 'socket-server'),
        default=None, help='Communication mode for remote environment execution of parallelized'
                           'environment execution'
    )
    parser.add_argument(
        '--blocking', action='store_true', help='Remote environments should be blocking'
    )
    parser.add_argument(
        '--host', type=str, default=None,
        help='Socket server hostname(s) or IP address(es), single value or comma-separated list'
    )
    parser.add_argument(
        '--port', type=str, default=None,
        help='Socket server port(s), single value or comma-separated list, increasing sequence if'
             'single host and port given'
    )
    # Runner arguments
    parser.add_argument(
        '-v', '--evaluation', action='store_true',
        help='Run environment (last if multiple) in evaluation mode'
    )
    parser.add_argument('-p', '--episodes', type=int, default=None, help='Number of episodes')
    parser.add_argument('-t', '--timesteps', type=int, default=None, help='Number of timesteps')
    parser.add_argument('-u', '--updates', type=int, default=None, help='Number of agent updates')
    parser.add_argument(
        '--mean-horizon', type=int, default=1,
        help='Number of episodes progress bar values and evaluation score are averaged over'
    )
    parser.add_argument(
        '--save-best-agent', type=str, default=None,
        help='Directory to save the best version of the agent according to the evaluation score'
    )
    # Logging arguments
    parser.add_argument('-r', '--repeat', type=int, default=1, help='Number of repetitions')
    parser.add_argument(
        '--path', type=str, default=None,
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

        def callback(r, p):
            rewards[r.episodes - 1].append(r.episode_rewards[-1])
            timesteps[r.episodes - 1].append(r.episode_timesteps[-1])
            seconds[r.episodes - 1].append(r.episode_seconds[-1])
            agent_seconds[r.episodes - 1].append(r.episode_agent_seconds[-1])
            return True

    if args.environment is None:
        environment = None
    else:
        environment = dict(environment=args.environment)
    if args.level is not None:
        environment['level'] = args.level
    if args.visualize:
        environment['visualize'] = True
    if args.visualize_directory is not None:
        environment['visualize_directory'] = args.visualize_directory

    if args.host is not None and ',' in args.host:
        args.host = args.host.split(',')
    if args.port is not None and ',' in args.port:
        args.port = [int(x) for x in args.port.split(',')]
    elif args.port is not None:
        args.port = int(args.port)

    if args.remote == 'socket-server':
        Environment.create(
            environment=environment, max_episode_timesteps=args.max_episode_timesteps,
            remote=args.remote, port=args.port
        )
        return

    if args.agent is None:
        agent = None
    else:
        agent = dict(agent=args.agent)
    if args.network is not None:
        agent['network'] = args.network

    for _ in range(args.repeat):
        runner = Runner(
            agent=agent, environment=environment, max_episode_timesteps=args.max_episode_timesteps,
            evaluation=args.evaluation, num_parallel=args.num_parallel, remote=args.remote,
            blocking=args.blocking, host=args.host, port=args.port
        )
        runner.run(
            num_episodes=args.episodes, num_timesteps=args.timesteps, num_updates=args.updates,
            batch_agent_calls=args.batch_agent_calls, sync_timesteps=args.sync_timesteps,
            sync_episodes=args.sync_episodes, callback=callback, mean_horizon=args.mean_horizon,
            save_best_agent=args.save_best_agent
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
                ), cls=NumpyJSONEncoder)
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
