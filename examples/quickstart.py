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

from tensorforce import Runner


def main():
    # OpenAI-Gym environment specification
    environment = dict(environment='gym', level='CartPole-v1')
    # or: environment = Environment.create(
    #         environment='gym', level='CartPole-v1', max_episode_timesteps=500)

    # PPO agent specification
    agent = dict(
        agent='ppo',
        # Automatically configured network
        network='auto',
        # PPO optimization parameters
        batch_size=10, update_frequency=2, learning_rate=3e-4, multi_step=10,
        subsampling_fraction=0.33,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
        # Baseline network and optimizer
        baseline=dict(type='auto', size=32, depth=1),
        baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # Preprocessing
        state_preprocessing='linear_normalization', reward_preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Default additional config values
        config=None,
        # Save agent every 10 updates and keep the 5 most recent checkpoints
        saver=dict(directory='model', frequency=10, max_checkpoints=5),
        # Log all available Tensorboard summaries
        summarizer=dict(directory='summaries', summaries='all'),
        # Do not record agent-environment interaction trace
        recorder=None
    )
    # or: Agent.create(agent='ppo', environment=environment, ...)
    # with additional argument "environment" and, if applicable, "parallel_interactions"

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment, max_episode_timesteps=500)

    # Train for 200 episodes
    runner.run(num_episodes=200)
    runner.close()

    # plus agent.close() and environment.close() if created separately


if __name__ == '__main__':
    main()
