# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from tensorforce import TensorforceError
from tensorforce.environments import Environment


class OpenAIGym(Environment):
    """
    [OpenAI Gym](https://gym.openai.com/) environment adapter (specification key: `gym`,
    `openai_gym`).

    May require:
    ```bash
    pip install gym[all]
    ```

    Args:
        level (string): Gym id
            (<span style="color:#C00000"><b>required</b></span>).
        visualize (bool): Whether to visualize interaction
            (<span style="color:#00C000"><b>default</b></span>: false).
        monitor_directory (string): Monitor output directory
            (<span style="color:#00C000"><b>default</b></span>: none).
        kwargs: Gym environment arguments.
    """

    @classmethod
    def levels(cls):
        import gym

        return list(gym.envs.registry.env_specs)

    def __init__(self, level, visualize=False, monitor_directory=None, **kwargs):
        import gym
        import gym.wrappers

        assert level in self.__class__.levels()

        self.env_id = level
        self.visualize = visualize

        self.create_gym(**kwargs)
        if monitor_directory is not None:
            self.environment = gym.wrappers.Monitor(
                env=self.environment, directory=monitor_directory
            )

        self.states_spec = OpenAIGym.specs_from_gym_space(
            space=self.environment.observation_space, ignore_value_bounds=True  # TODO: not ignore?
        )
        self.actions_spec = OpenAIGym.specs_from_gym_space(
            space=self.environment.action_space, ignore_value_bounds=False
        )

    def create_gym(self, **kwargs):
        import gym

        self.environment = gym.make(id=self.env_id, **kwargs)

    def __str__(self):
        return super().__str__() + '({})'.format(self.env_id)

    def states(self):
        return self.states_spec

    def actions(self):
        return self.actions_spec

    def max_episode_timesteps(self):
        if hasattr(self.environment, '_max_episode_steps'):
            return self.environment._max_episode_steps
        else:
            return super().max_episode_timesteps()

    def close(self):
        self.environment.close()
        self.environment = None

    def reset(self):
        import gym.wrappers

        if isinstance(self.environment, gym.wrappers.Monitor):
            self.environment.stats_recorder.done = True
        states = self.environment.reset()
        return OpenAIGym.flatten_state(state=states)

    def execute(self, actions):
        if self.visualize:
            self.environment.render()
        actions = OpenAIGym.unflatten_action(action=actions)
        states, reward, terminal, _ = self.environment.step(actions)
        return OpenAIGym.flatten_state(state=states), terminal, reward

    @staticmethod
    def specs_from_gym_space(space, ignore_value_bounds):
        import gym
        if isinstance(space, gym.spaces.Discrete):
            return dict(type='int', shape=(), num_values=space.n)

        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(type='bool', shape=space.n)

        elif isinstance(space, gym.spaces.MultiDiscrete):
            num_discrete_space = len(space.nvec)
            if (space.nvec == space.nvec[0]).all():
                return dict(type='int', shape=num_discrete_space, num_values=space.nvec[0])
            else:
                specs = dict()
                for n in range(num_discrete_space):
                    specs['gymmdc{}'.format(n)] = dict(
                        type='int', shape=(), num_values=space.nvec[n]
                    )
                return specs

        elif isinstance(space, gym.spaces.Box):
            if ignore_value_bounds:
                return dict(type='float', shape=space.shape)
            elif (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(
                    type='float', shape=space.shape, min_value=space.low[0],
                    max_value=space.high[0]
                )
            else:
                specs = dict()
                low = space.low.flatten()
                high = space.high.flatten()
                for n in range(low.shape[0]):
                    specs['gymbox{}'.format(n)] = dict(
                        type='float', shape=(), min_value=low[n], max_value=high[n]
                    )
                return specs

        elif isinstance(space, gym.spaces.Tuple):
            specs = dict()
            n = 0
            for n, space in enumerate(space.spaces):
                spec = OpenAIGym.specs_from_gym_space(
                    space=space, ignore_value_bounds=ignore_value_bounds
                )
                if 'type' in spec:
                    specs['gymtpl{}'.format(n)] = spec
                else:
                    for name, spec in spec.items():
                        specs['gymtpl{}-{}'.format(n, name)] = spec
            return specs

        elif isinstance(space, gym.spaces.Dict):
            specs = dict()
            for space_name, space in space.spaces.items():
                spec = OpenAIGym.specs_from_gym_space(
                    space=space, ignore_value_bounds=ignore_value_bounds
                )
                if 'type' in spec:
                    specs[space_name] = spec
                else:
                    for name, spec in spec.items():
                        specs['{}-{}'.format(space_name, name)] = spec
            return specs

        else:
            raise TensorforceError('Unknown Gym space.')

    @staticmethod
    def flatten_state(state):
        if isinstance(state, tuple):
            states = dict()
            for n, state in enumerate(state):
                state = OpenAIGym.flatten_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['gymtpl{}-{}'.format(n, name)] = state
                else:
                    states['gymtpl{}'.format(n)] = state
            return states

        elif isinstance(state, dict):
            states = dict()
            for state_name, state in state.items():
                state = OpenAIGym.flatten_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['{}-{}'.format(state_name, name)] = state
                else:
                    states[state_name] = state
            return states

        elif np.isinf(state).any() or np.isnan(state).any():
            raise TensorforceError("State contains inf or nan.")

        else:
            return state

    @staticmethod
    def unflatten_action(action):
        if not isinstance(action, dict):
            if np.isinf(action).any() or np.isnan(action).any():
                raise TensorforceError("Action contains inf or nan.")

            return action

        elif all(name.startswith('gymmdc') for name in action) or \
                all(name.startswith('gymbox') for name in action) or \
                all(name.startswith('gymtpl') for name in action):
            space_type = next(iter(action))[:6]
            actions = list()
            n = 0
            while True:
                if any(name.startswith(space_type + str(n) + '-') for name in action):
                    inner_action = {
                        name[name.index('-') + 1:] for name, inner_action in action.items()
                        if name.startswith(space_type + str(n))
                    }
                    actions.append(OpenAIGym.unflatten_action(action=inner_action))
                elif any(name == space_type + str(n) for name in action):
                    actions.append(OpenAIGym.unflatten_action(action=action[space_type + str(n)]))
                else:
                    break
                n += 1
            return tuple(actions)

        else:
            actions = dict()
            for name, action in action.items():
                if '-' in name:
                    name, inner_name = name.split('-', 1)
                    if name not in actions:
                        actions[name] = dict()
                    actions[name][inner_name] = action
                else:
                    actions[name] = action
            for name, action in actions.items():
                if isinstance(action, dict):
                    actions[name] = OpenAIGym.unflatten_action(action=action)
            return actions
