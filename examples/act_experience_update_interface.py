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

from tensorforce import Agent, Environment


def main():
    environment = Environment.create(environment='benchmarks/configs/cartpole.json')
    agent = Agent.create(agent='benchmarks/configs/ppo.json', environment=environment)

    # Train for 100 episodes
    for episode in range(100):

        # Record episode experience
        episode_states = list()
        episode_internals = list()
        episode_actions = list()
        episode_terminal = list()
        episode_reward = list()

        # Episode using independent-act and agent.intial_internals()
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        sum_reward = 0.0
        while not terminal:
            episode_states.append(states)
            episode_internals.append(internals)
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            episode_actions.append(actions)
            states, terminal, reward = environment.execute(actions=actions)
            episode_terminal.append(terminal)
            episode_reward.append(reward)
            sum_reward += reward
        print('Episode {}: {}'.format(episode, sum_reward))

        # Feed recorded experience to agent
        agent.experience(
            states=episode_states, internals=episode_internals, actions=episode_actions,
            terminal=episode_terminal, reward=episode_reward
        )

        # Perform update
        agent.update()

    # Evaluate for 100 episodes
    sum_rewards = 0.0
    for _ in range(100):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
    print('Mean evaluation return:', sum_rewards / 100.0)

    # Close agent and environment
    agent.close()
    environment.close()


if __name__ == '__main__':
    main()
