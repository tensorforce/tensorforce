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

        # Episode using act and observe
        states = environment.reset()
        terminal = False
        sum_rewards = 0.0
        num_updates = 0
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            num_updates += agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
        print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

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
