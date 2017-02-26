# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
OpenAI gym execution
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import argparse

from tensorforce.config import Config, create_config
from tensorforce.execution.distributed_runner import DistributedRunner
from tensorforce.external.openai_gym import OpenAIGymEnvironment
from tensorforce.util.experiment_util import build_preprocessing_stack


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

    print("Starting distributed agent for OpenAI Gym '{gym_id}'".format(gym_id=args.gym_id))
    print("Config:")
    print(config)
    max_global_steps = 10000000
    runner = DistributedRunner(agent_type=args.agent, agent_config=config, n_agents=1,
                               environment=env, preprocessor=stack, repeat_actions=args.repeat_actions,
                               episodes=args.episodes, max_timesteps=max_global_steps, local_steps=10, n_param_servers=1)
    runner.run()


if __name__ == '__main__':
    main()
