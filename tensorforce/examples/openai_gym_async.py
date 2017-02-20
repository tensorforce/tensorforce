# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
OpenAI gym execution
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
from six.moves import xrange

import numpy as np

from tensorforce.config import Config, create_config
from tensorforce.execution.distributed_runner import DistributedRunner
from tensorforce.external.openai_gym import OpenAIGymEnvironment
from tensorforce.util.experiment_util import build_preprocessing_stack
from tensorforce.util.agent_util import create_agent, get_default_config
from tensorforce import preprocessing


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-a', '--agent', default='DQNAgent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file",
                        default='examples/configs/dqn_agent.json')
    parser.add_argument('-n', '--network-config', help="Network configuration file",
                        default='examples/configs/dqn_network.json')
    parser.add_argument('-e', '--episodes', type=int, default=10000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-r', '--repeat-actions', type=int, default=4, help="???")
    parser.add_argument('-m', '--monitor', help="Save results to this file")
    args = parser.parse_args()

    env = OpenAIGymEnvironment(args.gym_id)

    config = Config({
        'repeat_actions': 1,
        'actions': env.actions,
        'action_shape': env.action_shape,
        'state_shape': env.state_shape
    })

    if args.agent_config:
        config.read_json(args.agent_config)
    if args.network_config:
        config.read_json(args.network_config)

    report_episodes = args.episodes / 10

    preprocessing_config = config.get('preprocessing')

    if preprocessing_config:
        stack = build_preprocessing_stack(preprocessing_config)
        config.state_shape = stack.shape(config.state_shape)
    else:
        stack = None

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
            print("Total reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 500 rewards: {}".format(np.mean(r.episode_rewards[-500:])))
            print("Average of last 100 rewards: {}".format(np.mean(r.episode_rewards[-100:])))
        return True

    print("Starting {agent_type} for OpenAI Gym '{gym_id}'".format(agent_type=args.agent, gym_id=args.gym_id))
    print("Config:")
    print(config)
    # if args.monitor:
    #     for environment in environments:
    #         environment.gym.monitor.start(args.monitor)
    #         environment.gym.monitor.configure(video_callable=lambda count: False)  # count % 500 == 0)

    runner = DistributedRunner(agent_type=args.agent, agent_config=config, n_agents=2,
                               environment=env, preprocessor=stack, repeat_actions=args.repeat_actions,
                               episodes=args.episodes, max_timesteps=args.max_timesteps, n_param_servers=1)
    runner.run()

    # if args.monitor:
    #     for environment in environments:
    #         environment.gym.monitor.close()
    #print("Learning finished. Total episodes: {ep}".format(ep=execution.episode + 1))
    # for environment in environments:
    #     environment.close()


if __name__ == '__main__':
    main()
