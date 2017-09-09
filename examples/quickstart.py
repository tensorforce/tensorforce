# Copyright 2017 reinforce.io. All Rights Reserved.
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

"""
Quick start example.
"""

import numpy as np

from tensorforce import Configuration
from tensorforce.agents import TRPOAgent, PPOAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Create an OpenAIgym environment
env = OpenAIGym('CartPole-v0')

# Create a Trust Region Policy Optimization agent
agent = PPOAgent(
    Configuration(
    log_level='info',
    batch_size=4000,

    # max_kl_divergence=0.1,
    # cg_iterations=20,
    # cg_damping=0.001,
    # ls_max_backtracks=10,
    # ls_accept_ratio=0.9,
    # ls_override=False,

    learning_rate=0.001,
    entropy_penalty=0.01,
    epochs=5,
    optimizer_batch_size=512,
    loss_clipping=0.2,
    normalize_advantage=False,
    baseline=dict(
        type="mlp",
        sizes=[32, 32],
        epochs=1,
        update_batch_size=512,
        learning_rate=0.01
    ),
    states=env.states,
    actions=env.actions,
    network=layered_network_builder([
        dict(type='dense', size=32, activation='tanh'),
        dict(type='dense', size=32, activation='tanh')
    ])
))

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=3000, max_timesteps=200, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(ep=runner.episode,
                                                                                                   ar=np.mean(
                                                                                                       runner.episode_rewards[
                                                                                                       -100:])))
