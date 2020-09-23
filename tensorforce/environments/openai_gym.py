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

import importlib

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
        min_value (float): Lower bound clipping for otherwise unbounded state values
            (<span style="color:#00C000"><b>default</b></span>: no clipping).
        max_value (float): Upper bound clipping for otherwise unbounded state values
            (<span style="color:#00C000"><b>default</b></span>: no clipping).
        terminal_reward (float): Additional reward for early termination, if otherwise
            indistinguishable from termination due to maximum number of timesteps
            (<span style="color:#00C000"><b>default</b></span>: Gym default).
        reward_threshold (float): Gym environment argument, the reward threshold before the task is
            considered solved
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
    def create_level(cls, level, max_episode_steps, reward_threshold, **kwargs):
        import gym

        requires_register = False

        # Find level
        if level not in gym.envs.registry.env_specs:
            if max_episode_steps is None:  # interpret as false if level does not exist
                max_episode_steps = False
            env_specs = list(gym.envs.registry.env_specs)
            if level + '-v0' in gym.envs.registry.env_specs:
                env_specs.insert(0, level + '-v0')
            search = level
            level = None
            for name in env_specs:
                if search == name[:name.rindex('-v')]:
                    if level is None:
                        level = name
                    if max_episode_steps is False and \
                            gym.envs.registry.env_specs[name].max_episode_steps is not None:
                        continue
                    elif max_episode_steps != gym.envs.registry.env_specs[name].max_episode_steps:
                        continue
                    level = name
                    break
            else:
                if level is None:
                    raise TensorforceError.value(name='OpenAIGym', argument='level', value=level)
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

        if max_episode_steps is False:
            max_episode_steps = None

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
                nondeterministic=nondeterministic, max_episode_steps=max_episode_steps,
                kwargs=_kwargs
            )
            assert level in cls.levels()

        return gym.make(id=level, **kwargs), max_episode_steps

    def __init__(
        self, level, visualize=False, import_modules=None, min_value=None, max_value=None,
        terminal_reward=0.0, reward_threshold=None, drop_states_indices=None,
        visualize_directory=None, **kwargs
    ):
        super().__init__()

        import gym
        import gym.wrappers

        if import_modules is None:
            pass
        elif isinstance(import_modules, str):
            importlib.import_module(name=import_modules)
        elif isinstance(import_modules, (list, tuple)):
            for module in import_modules:
                importlib.import_module(name=module)

        self.level = level
        self.visualize = visualize
        self.terminal_reward = terminal_reward

        if isinstance(level, gym.Env):
            self.environment = self.level
            self.level = self.level.__class__.__name__
            self._max_episode_timesteps = None
        elif isinstance(level, type) and issubclass(level, gym.Env):
            self.environment = self.level(**kwargs)
            self.level = self.level.__class__.__name__
            self._max_episode_timesteps = None
        else:
            self.environment, self._max_episode_timesteps = self.__class__.create_level(
                level=self.level, max_episode_steps=None, reward_threshold=reward_threshold,
                **kwargs
            )

        if visualize_directory is not None:
            self.environment = gym.wrappers.Monitor(
                env=self.environment, directory=visualize_directory
            )

        self.min_value = min_value
        self.max_value = max_value
        if min_value is not None:
            if max_value is None:
                raise TensorforceError.required(name='OpenAIGym', argument='max_value')
            self.states_spec = OpenAIGym.specs_from_gym_space(
                space=self.environment.observation_space, min_value=min_value, max_value=max_value
            )
        elif max_value is not None:
            raise TensorforceError.required(name='OpenAIGym', argument='min_value')
        else:
            self.states_spec = OpenAIGym.specs_from_gym_space(
                space=self.environment.observation_space, allow_infinite_box_bounds=True
            )

        if drop_states_indices is None:
            self.drop_states_indices = None
        else:
            assert 'shape' in self.states_spec
            self.drop_states_indices = sorted(drop_states_indices)
            assert len(self.states_spec['shape']) == 1
            num_dropped = len(self.drop_states_indices)
            self.states_spec['shape'] = (self.states_spec['shape'][0] - num_dropped,)

        self.actions_spec = OpenAIGym.specs_from_gym_space(space=self.environment.action_space)

    def __str__(self):
        return super().__str__() + '({})'.format(self.level)

    def states(self):
        return self.states_spec

    def actions(self):
        return self.actions_spec

    def max_episode_timesteps(self):
        return self._max_episode_timesteps

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
        if self.min_value is not None:
            states = np.clip(states, self.states_spec['min_value'], self.states_spec['max_value'])
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
        if self._max_episode_timesteps is not None and self.timestep == self._max_episode_timesteps:
            assert terminal
            terminal = 2
        elif terminal:
            assert self._max_episode_timesteps is None or \
                self.timestep < self._max_episode_timesteps
            reward += self.terminal_reward
            terminal = 1
        else:
            terminal = 0
        states = OpenAIGym.flatten_state(state=states, states_spec=self.states_spec)
        if self.min_value is not None:
            states = np.clip(states, self.states_spec['min_value'], self.states_spec['max_value'])
        if self.drop_states_indices is not None:
            for index in reversed(self.drop_states_indices):
                states = np.concatenate([states[:index], states[index + 1:]])
        return states, terminal, reward

    @staticmethod
    def specs_from_gym_space(
        space, allow_infinite_box_bounds=False, min_value=None, max_value=None
    ):
        import gym

        if isinstance(space, gym.spaces.Discrete):
            return dict(type='int', shape=(), num_values=space.n)

        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(type='bool', shape=space.n)

        elif isinstance(space, gym.spaces.MultiDiscrete):
            if (space.nvec == space.nvec.item(0)).all():
                return dict(type='int', shape=space.nvec.shape, num_values=space.nvec.item(0))
            else:
                specs = dict()
                nvec = space.nvec.flatten()
                shape = '_'.join(str(x) for x in space.nvec.shape)
                for n in range(nvec.shape[0]):
                    specs['gymmdc{}_{}'.format(n, shape)] = dict(
                        type='int', shape=(), num_values=nvec[n]
                    )
                return specs

        elif isinstance(space, gym.spaces.Box):
            spec = dict(type='float', shape=space.shape)

            if (space.low == space.low.item(0)).all():
                _min_value = float(space.low.item(0))
                if _min_value > -10e7:
                    spec['min_value'] = _min_value
                else:
                    spec['min_value'] = min_value
            elif allow_infinite_box_bounds:
                _min_value = np.where(space.low <= -10e7, -np.inf, space.low)
                spec['min_value'] = _min_value.astype(util.np_dtype(dtype='float'))
            elif (space.low > -10e7).all():
                spec['min_value'] = space.low.astype(util.np_dtype(dtype='float'))
            elif min_value is None:
                raise TensorforceError("Invalid infinite box bounds")
            else:
                _min_value = np.where(space.low <= -10e7, min_value, space.low)
                spec['min_value'] = _min_value.astype(util.np_dtype(dtype='float'))

            if spec is None:
                pass
            elif (space.high == space.high.item(0)).all():
                _max_value = float(space.high.item(0))
                if _max_value < 10e7:
                    spec['max_value'] = _max_value
                else:
                    spec['max_value'] = max_value
            elif allow_infinite_box_bounds:
                _max_value = np.where(space.high >= 10e7, np.inf, space.high)
                spec['max_value'] = _max_value.astype(util.np_dtype(dtype='float'))
            elif (space.high < 10e7).all():
                spec['max_value'] = space.high.astype(util.np_dtype(dtype='float'))
            elif max_value is None:
                raise TensorforceError("OpenAIGym: Invalid infinite box bounds")
            else:
                _max_value = np.where(space.high >= 10e7, max_value, space.high)
                spec['max_value'] = _max_value.astype(util.np_dtype(dtype='float'))

            if spec is None:
                specs = dict()
                low = space.low.flatten()
                high = space.high.flatten()
                shape = '_'.join(str(x) for x in space.low.shape)
                for n in range(low.shape[0]):
                    spec = dict(type='float', shape=())
                    if low[n] > -10e7:
                        spec['min_value'] = float(low[n])
                    if high[n] < 10e7:
                        spec['max_value'] = float(high[n])
                    specs['gymbox{}_{}'.format(n, shape)] = spec
                return specs
            else:
                return spec

        elif isinstance(space, gym.spaces.Tuple):
            specs = dict()
            n = 0
            for n, space in enumerate(space.spaces):
                spec = OpenAIGym.specs_from_gym_space(
                    space=space, allow_infinite_box_bounds=allow_infinite_box_bounds
                )
                if 'type' in spec:
                    specs['gymtpl{}'.format(n)] = spec
                else:
                    for name, spec in spec.items():
                        specs['gymtpl{}_{}'.format(n, name)] = spec
            return specs

        elif isinstance(space, gym.spaces.Dict):
            specs = dict()
            for space_name, space in space.spaces.items():
                spec = OpenAIGym.specs_from_gym_space(
                    space=space, allow_infinite_box_bounds=allow_infinite_box_bounds
                )
                if 'type' in spec:
                    specs[space_name] = spec
                else:
                    for name, spec in spec.items():
                        specs['{}_{}'.format(space_name, name)] = spec
            return specs

        else:
            raise TensorforceError('Unknown Gym space.')

    @staticmethod
    def flatten_state(state, states_spec):
        if isinstance(state, tuple):
            states = dict()
            for n, state in enumerate(state):
                if 'gymtpl{}'.format(n) in states_spec:
                    spec = states_spec['gymtpl{}'.format(n)]
                else:
                    spec = None
                    for name in states_spec:
                        if name.startswith('gymtpl{}_'.format(n)):
                            assert spec is None
                            spec = states_spec[name]
                    assert spec is not None
                state = OpenAIGym.flatten_state(state=state, states_spec=spec)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['gymtpl{}_{}'.format(n, name)] = state
                else:
                    states['gymtpl{}'.format(n)] = state
            return states

        elif isinstance(state, dict):
            states = dict()
            for state_name, state in state.items():
                if state_name in states_spec:
                    spec = states_spec[state_name]
                else:
                    spec = None
                    for name in states_spec:
                        if name.startswith('{}_'.format(state_name)):
                            assert spec is None
                            spec = states_spec[name]
                    assert spec is not None
                state = OpenAIGym.flatten_state(state=state, states_spec=spec)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['{}_{}'.format(state_name, name)] = state
                else:
                    states[state_name] = state
            return states

        elif np.isinf(state).any() or np.isnan(state).any():
            raise TensorforceError("State contains inf or nan.")

        elif 'gymbox0' in states_spec:
            states = dict()
            state = state.flatten()
            shape = '_'.join(str(x) for x in state.shape)
            for n in range(state.shape[0]):
                states['gymbox{}_{}'.format(n, shape)] = state[n]
            return states

        elif 'gymmdc0' in states_spec:
            states = dict()
            state = state.flatten()
            shape = '_'.join(str(x) for x in state.shape)
            for n in range(state.shape[0]):
                states['gymmdc{}_{}'.format(n, shape)] = state[n]
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
                if any(name.startswith(space_type + str(n) + '_') for name in action):
                    inner_action = [
                        value for name, value in action.items()
                        if name.startswith(space_type + str(n))
                    ]
                    assert len(inner_action) == 1
                    actions.append(OpenAIGym.unflatten_action(action=inner_action[0]))
                elif any(name == space_type + str(n) for name in action):
                    actions.append(OpenAIGym.unflatten_action(action=action[space_type + str(n)]))
                else:
                    break
                n += 1
            if all(name.startswith('gymmdc') for name in action) or \
                    all(name.startswith('gymbox') for name in action):
                name = next(iter(action))
                shape = tuple(int(x) for x in name[name.index('_') + 1:].split('_'))
                return np.array(actions).reshape(shape)
            else:
                return tuple(actions)

        else:
            actions = dict()
            for name, action in action.items():
                if '_' in name:
                    name, inner_name = name.split('_', 1)
                    if name not in actions:
                        actions[name] = dict()
                    actions[name][inner_name] = action
                else:
                    actions[name] = action
            for name, action in actions.items():
                if isinstance(action, dict):
                    actions[name] = OpenAIGym.unflatten_action(action=action)
            return actions
