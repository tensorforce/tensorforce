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

import numpy as np
import tensorflow as tf

from tensorforce import Environment, Runner


def main():
    # Train agent
    environment = Environment.create(environment='benchmarks/configs/cartpole.json')
    runner = Runner(agent='benchmarks/configs/ppo.json', environment=environment)
    runner.run(num_episodes=100)

    # Save agent SavedModel
    runner.agent.save(directory='saved-model', format='saved-model')
    runner.close()

    # Model serving, potentially using different programming language etc

    # Load agent SavedModel
    agent = tf.saved_model.load(export_dir='saved-model')

    # Evaluate for 100 episodes
    sum_rewards = 0.0
    for _ in range(100):
        states = environment.reset()

        # Required in case of internal states:
        # internals = agent.initial_internals()
        # internals = recursive_map(batch, internals)

        terminal = False
        while not terminal:

            states = batch(states)
            # Required in case of nested states:
            # states = recursive_map(batch, states)

            auxiliaries = dict(mask=np.ones(shape=(1, 2), dtype=bool))
            deterministic = True

            actions = agent.act(states, auxiliaries, deterministic)
            # Required in case of internal states:
            # actions_internals = agent.act(states, internals, auxiliaries, deterministic)
            # actions, internals = actions_internals['actions'], actions_internals['internals']

            actions = unbatch(actions)
            # Required in case of nested actions:
            # actions = recursive_map(unbatch, actions)

            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward

    print('Mean evaluation return:', sum_rewards / 100.0)
    environment.close()


# Batch inputs
def batch(x):
    return np.expand_dims(x, axis=0)


# Unbatch outputs
def unbatch(x):
    if isinstance(x, tf.Tensor):  # TF tensor to NumPy array
        x = x.numpy()
    if x.shape == (1,):  # Singleton array to Python value
        return x.item()
    else:
        return np.squeeze(x, axis=0)


# Apply function to leaf values in nested dict
# (required for nested states/actions)
def recursive_map(function, dictionary):
    mapped = dict()
    for key, value in dictionary.items():
        if isinstance(value, dict):
            mapped[key] = recursive_map(function, value)
        else:
            mapped[key] = function(value)
    return mapped


if __name__ == '__main__':
    main()
