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
    num_parallel = 8
    environment = Environment.create(environment='custom_cartpole', max_episode_timesteps=500)
    agent = Agent.create(
        agent='benchmarks/configs/ppo.json', environment=environment,
        parallel_interactions=num_parallel
    )

    # Train for 100 episodes
    for episode in range(0, 100, num_parallel):

        # Episode using act and observe
        parallel, states = environment.reset(num_parallel=num_parallel)
        terminal = (parallel < 0)  # all false
        sum_rewards = 0.0
        num_updates = 0
        while not terminal.all():
            actions = agent.act(states=states, parallel=parallel)
            next_parallel, states, terminal, reward = environment.execute(actions=actions)
            num_updates += agent.observe(terminal=terminal, reward=reward, parallel=parallel)
            parallel = next_parallel
            sum_rewards += reward.sum()
        print('Episode {}: return={} updates={}'.format(
            episode, sum_rewards / num_parallel, num_updates
        ))

    # Evaluate for 100 episodes
    num_parallel = 4
    num_episodes = 100
    sum_rewards = 0.0
    for _ in range(0, num_episodes, num_parallel):
        parallel, states = environment.reset(num_parallel=num_parallel)
        internals = agent.initial_internals()
        internals = [internals for _ in range(num_parallel)]
        terminal = (parallel < 0)  # all false
        while not terminal.all():
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            _, states, terminal, reward = environment.execute(actions=actions)
            internals = [internal for internal, term in zip(internals, terminal) if not term]
            sum_rewards += reward.sum()
    print('Mean evaluation return:', sum_rewards / num_episodes)

    # Close agent and environment
    agent.close()
    environment.close()


if __name__ == '__main__':
    main()
