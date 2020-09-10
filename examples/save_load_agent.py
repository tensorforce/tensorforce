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
    # OpenAI-Gym environment initialization
    environment = Environment.create(environment='benchmarks/configs/cartpole.json')

    # PPO agent initialization
    agent = Agent.create(
        agent='benchmarks/configs/ppo.json', environment=environment,
        # Option 1: Saver - save agent periodically every 10 updates
        # and keep the 5 most recent checkpoints
        saver=dict(directory='model-checkpoint', frequency=10, max_checkpoints=5),
    )

    # Runner initialization
    runner = Runner(agent=agent, environment=environment)

    # Training
    runner.run(num_episodes=100)
    runner.close()

    # Option 2: Explicit save
    # (format: 'numpy' or 'hdf5' store only weights, 'checkpoint' stores full TensorFlow model,
    # agent argument saver, specified above, uses 'checkpoint')
    agent.save(directory='model-numpy', format='numpy', append='episodes')

    # Close agent separately, since created separately
    agent.close()

    # Load agent TensorFlow checkpoint
    agent = Agent.load(directory='model-checkpoint', format='checkpoint', environment=environment)
    runner = Runner(agent=agent, environment=environment)
    runner.run(num_episodes=100, evaluation=True)
    runner.close()
    agent.close()

    # Load agent NumPy weights
    agent = Agent.load(directory='model-numpy', format='numpy', environment=environment)
    runner = Runner(agent=agent, environment=environment)
    runner.run(num_episodes=100, evaluation=True)
    runner.close()
    agent.close()

    # Close environment separately, since created separately
    environment.close()


if __name__ == '__main__':
    main()
