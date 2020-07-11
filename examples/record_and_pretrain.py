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

from tensorforce import Agent, Environment, Runner


def main():
    # Start recording traces after the first 100 episodes -- by then, the agent
    # has solved the environment
    runner = Runner(
        agent=dict(
            agent='benchmarks/configs/ppo.json',
            recorder=dict(directory='ppo-traces', start=80)
        ), environment='benchmarks/configs/cartpole.json'
    )
    runner.run(num_episodes=100)
    runner.close()

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


if __name__ == '__main__':
    main()
