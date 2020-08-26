# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

import os

import numpy as np

from tensorforce import Agent, Environment, Runner


def main():
    # Record experience traces
    record_ppo_config(directory='ppo-traces')
    # Alternatively:
    # record_custom_act_function(directory='ppo-traces')
    # write_custom_recording_file(directory='ppo-traces')

    # Pretrain a new agent on the recorded traces: for 30 iterations, feed the
    # experience of one episode to the agent and subsequently perform one update
    environment = Environment.create(environment='benchmarks/configs/cartpole.json')
    agent = Agent.create(agent='benchmarks/configs/ppo.json', environment=environment)
    agent.pretrain(directory='ppo-traces', num_iterations=30, num_traces=1, num_updates=1)

    # Evaluate the pretrained agent
    runner = Runner(agent=agent, environment=environment)
    runner.run(num_episodes=100, evaluation=True)
    runner.close()

    # Close agent and environment
    agent.close()
    environment.close()


def record_ppo_config(directory):
    # Start recording traces after 80 episodes -- by then, the environment is solved
    runner = Runner(
        agent=dict(
            agent='benchmarks/configs/ppo.json',
            recorder=dict(directory=directory, start=80)
        ), environment='benchmarks/configs/cartpole.json'
    )
    runner.run(num_episodes=100)
    runner.close()


def record_custom_act_function(directory):
    # Trivial custom act function
    def fn_act(states):
        return int(states[2] < 0.0)

    # Record 20 episodes
    runner = Runner(
        agent=dict(agent=fn_act, recorder=dict(directory=directory)),
        environment='benchmarks/configs/cartpole.json'
    )
    # or: agent = Agent.create(agent=fn_act, recorder=dict(directory=directory))
    runner.run(num_episodes=20)
    runner.close()


def write_custom_recording_file(directory):
    # Start recording traces after 80 episodes -- by then, the environment is solved
    environment = Environment.create(environment='benchmarks/configs/cartpole.json')
    agent = Agent.create(agent='benchmarks/configs/ppo.json', environment=environment)
    runner = Runner(agent=agent, environment=environment)
    runner.run(num_episodes=80)
    runner.close()

    # Record 20 episodes
    for episode in range(20):

        # Record episode experience
        episode_states = list()
        episode_actions = list()
        episode_terminal = list()
        episode_reward = list()

        # Evaluation episode
        states = environment.reset()
        terminal = False
        while not terminal:
            episode_states.append(states)
            actions = agent.act(states=states, independent=True, deterministic=True)
            episode_actions.append(actions)
            states, terminal, reward = environment.execute(actions=actions)
            episode_terminal.append(terminal)
            episode_reward.append(reward)

        # Write recorded episode trace to npz file
        np.savez_compressed(
            file=os.path.join(directory, 'trace-{:09d}.npz'.format(episode)),
            states=np.stack(episode_states, axis=0),
            actions=np.stack(episode_actions, axis=0),
            terminal=np.stack(episode_terminal, axis=0),
            reward=np.stack(episode_reward, axis=0)
        )


if __name__ == '__main__':
    main()
