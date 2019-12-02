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

from tensorforce import TensorforceError, util
from tensorforce.environments import Environment


class OpenAIGym(Environment):
    """
    [OpenAI Gym](https://gym.openai.com/) environment adapter (specification key: `gym`,
    `openai_gym`).

    May require:
    ```bash
    pip3 install gym
    pip3 install gym[all]
    ```

    Args:
        level (string | gym.Env): Gym id or instance
            (<span style="color:#C00000"><b>required</b></span>).
        visualize (bool): Whether to visualize interaction
            (<span style="color:#00C000"><b>default</b></span>: false).
        max_episode_steps (false | int > 0): Whether to terminate an episode after a while,
            and if so, maximum number of timesteps per episode
            (<span style="color:#00C000"><b>default</b></span>: Gym default).
        terminal_reward (float): Additional reward for early termination, if otherwise
            indistinguishable from termination due to maximum number of timesteps
            (<span style="color:#00C000"><b>default</b></span>: Gym default).
        reward_threshold (float): Gym environment argument, the reward threshold before the task is
            considered solved
            (<span style="color:#00C000"><b>default</b></span>: Gym default).
        tags (dict): Gym environment argument, a set of arbitrary key-value tags on this
            environment, including simple property=True tags
            (<span style="color:#00C000"><b>default</b></span>: Gym default).
        drop_states_indices (list[int]): Drop states indices
            (<span style="color:#00C000"><b>default</b></span>: none).
        visualize_directory (string): Visualization output directory
            (<span style="color:#00C000"><b>default</b></span>: none).
        kwargs: Additional Gym environment arguments.
    """

    @classmethod
    def levels(cls):
        import gym

        return list(gym.envs.registry.env_specs)

    @classmethod
    def create_level(cls, level, max_episode_steps, reward_threshold, tags, **kwargs):
        import gym

        requires_register = False

        # Find level
        if level not in gym.envs.registry.env_specs:
            if max_episode_steps is None:  # interpret as false if level does not exist
                max_episode_steps = False
            env_specs = list(gym.envs.registry.env_specs)
            if level + '-v0' in gym.envs.registry.env_specs:
                env_specs.insert(0, level + '-v0')
            for name in env_specs:
                if level == name[:name.rindex('-v')]:
                    if max_episode_steps is False and \
                            gym.envs.registry.env_specs[name].max_episode_steps is not None:
                        continue
                    elif max_episode_steps != gym.envs.registry.env_specs[name].max_episode_steps:
                        continue
                    level = name
                    break
            else:
                level = env_specs[0]
                requires_register = True
        assert level in cls.levels()

        # Check/update attributes
        if max_episode_steps is None:
            max_episode_steps = gym.envs.registry.env_specs[level].max_episode_steps
            if max_episode_steps is None:
                max_episode_steps = False
        elif max_episode_steps != gym.envs.registry.env_specs[level].max_episode_steps:
            if not (
                (max_episode_steps is False) and
                (gym.envs.registry.env_specs[level].max_episode_steps is None)
            ):
                requires_register = True
        if reward_threshold is None:
            reward_threshold = gym.envs.registry.env_specs[level].reward_threshold
        elif reward_threshold != gym.envs.registry.env_specs[level].reward_threshold:
            requires_register = True
        if tags is None:
            tags = dict(gym.envs.registry.env_specs[level].tags)
            if 'wrapper_config.TimeLimit.max_episode_steps' in tags and \
                    max_episode_steps is not None:
                tags.pop('wrapper_config.TimeLimit.max_episode_steps')
        elif tags != gym.envs.registry.env_specs[level].tags:
            requires_register = True

        # Modified specification
        if requires_register:
            entry_point = gym.envs.registry.env_specs[level].entry_point
            _kwargs = dict(gym.envs.registry.env_specs[level]._kwargs)
            nondeterministic = gym.envs.registry.env_specs[level].nondeterministic

            if '-v' in level and level[level.rindex('-v') + 2:].isdigit():
                version = int(level[level.rindex('-v') + 2:])
                level = level[:level.rindex('-v') + 2]
            else:
                version = -1
            while True:
                version += 1
                if level + str(version) not in gym.envs.registry.env_specs:
                    level = level + str(version)
                    break

            gym.register(
                id=level, entry_point=entry_point, reward_threshold=reward_threshold,
                kwargs=_kwargs, nondeterministic=nondeterministic, tags=tags,
                max_episode_steps=(None if max_episode_steps is False else max_episode_steps)
            )
            assert level in cls.levels()

        return gym.make(id=level, **kwargs), max_episode_steps

    def __init__(
        self, level, visualize=False, max_episode_steps=None, terminal_reward=0.0,
        reward_threshold=None, tags=None, drop_states_indices=None, visualize_directory=None,
        **kwargs
    ):
        super().__init__()

        import gym
        import gym.wrappers

        self.level = level
        self.visualize = visualize
        self.terminal_reward = terminal_reward

        if isinstance(level, gym.Env):
            self.environment = self.level
            self.level = self.level.__class__.__name__
            self.max_episode_steps = max_episode_steps
        else:
            self.environment, self.max_episode_steps = self.__class__.create_level(
                level=self.level, max_episode_steps=max_episode_steps,
                reward_threshold=reward_threshold, tags=tags, **kwargs
            )

        if visualize_directory is not None:
            self.environment = gym.wrappers.Monitor(
                env=self.environment, directory=visualize_directory
            )

        self.states_spec = OpenAIGym.specs_from_gym_space(
            space=self.environment.observation_space, ignore_value_bounds=True  # TODO: not ignore?
        )
        if drop_states_indices is None:
            self.drop_states_indices = None
        else:
            assert util.is_atomic_values_spec(values_spec=self.states_spec)
            self.drop_states_indices = sorted(drop_states_indices)
            assert len(self.states_spec['shape']) == 1
            num_dropped = len(self.drop_states_indices)
            self.states_spec['shape'] = (self.states_spec['shape'][0] - num_dropped,)

        self.actions_spec = OpenAIGym.specs_from_gym_space(
            space=self.environment.action_space, ignore_value_bounds=False
        )

    def __str__(self):
        return super().__str__() + '({})'.format(self.level)

    def states(self):
        return self.states_spec

    def actions(self):
        return self.actions_spec

    def max_episode_timesteps(self):
        if self.max_episode_steps is False:
            return super().max_episode_timesteps()
        else:
            return self.max_episode_steps

    def close(self):
        self.environment.close()
        self.environment = None

    def reset(self):
        import gym.wrappers

        if isinstance(self.environment, gym.wrappers.Monitor):
            self.environment.stats_recorder.done = True
        states = self.environment.reset()
        self.timestep = 0
        states = OpenAIGym.flatten_state(state=states, states_spec=self.states_spec)
        if self.drop_states_indices is not None:
            for index in reversed(self.drop_states_indices):
                states = np.concatenate([states[:index], states[index + 1:]])
        return states

    def execute(self, actions):
        if self.visualize:
            self.environment.render()
        actions = OpenAIGym.unflatten_action(action=actions)
        states, reward, terminal, _ = self.environment.step(actions)
        self.timestep += 1
        if self.timestep == self.max_episode_steps:
            assert terminal
            terminal = 2
        elif terminal:
            assert self.max_episode_steps is None or self.max_episode_steps is False or \
                self.timestep < self.max_episode_steps
            reward += self.terminal_reward
            terminal = 1
        else:
            terminal = 0
        states = OpenAIGym.flatten_state(state=states, states_spec=self.states_spec)
        if self.drop_states_indices is not None:
            for index in reversed(self.drop_states_indices):
                states = np.concatenate([states[:index], states[index + 1:]])
        return states, terminal, reward

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
    def flatten_state(state, states_spec):
        if isinstance(state, tuple):
            states = dict()
            for n, state in enumerate(state):
                if 'gymtpl{}-{}'.format(n, name) in states_spec:
                    spec = states_spec['gymtpl{}-{}'.format(n, name)]
                elif 'gymtpl{}'.format(n) in states_spec:
                    spec = states_spec['gymtpl{}'.format(n)]
                else:
                    raise TensorforceError.unexpected()
                state = OpenAIGym.flatten_state(state=state, states_spec=spec)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['gymtpl{}-{}'.format(n, name)] = state
                else:
                    states['gymtpl{}'.format(n)] = state
            return states

        elif isinstance(state, dict):
            states = dict()
            for state_name, state in state.items():
                if '{}-{}'.format(state_name, name) in states_spec:
                    spec = states_spec['{}-{}'.format(state_name, name)]
                elif state_name in states_spec:
                    spec = states_spec[state_name]
                else:
                    raise TensorforceError.unexpected()
                state = OpenAIGym.flatten_state(state=state, states_spec=spec)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['{}-{}'.format(state_name, name)] = state
                else:
                    states[state_name] = state
            return states

        elif np.isinf(state).any() or np.isnan(state).any():
            raise TensorforceError("State contains inf or nan.")

        elif 'gymbox0' in states_spec:
            states = dict()
            for n in range(state.shape[0]):
                states['gymbox{}'.format(n)] = state[n]
            return states

        elif 'gymmdc0' in states_spec:
            states = dict()
            for n in range(state.shape[0]):
                states['gymmdc{}'.format(n)] = state[n]
            return states

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
