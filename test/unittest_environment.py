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

from collections import OrderedDict
from random import randint, random

import numpy as np

from tensorforce import TensorforceError, util
from tensorforce.environments import Environment


class UnittestEnvironment(Environment):
    """
    Unit-test environment.
    """

    def __init__(self, states, actions, timestep_range):
        """
        Initializes a mock environment which is used for the unit-tests.

        Args:
            states: The state specification.
            actions: The action specification.
            timestep_range: The range of non-terminal timesteps.
        """
        super().__init__()

        self.states_spec = OrderedDict((name, states[name]) for name in sorted(states))
        self.actions_spec = OrderedDict((name, actions[name]) for name in sorted(actions))
        self.timestep_range = timestep_range

        self.random_states = self.__class__.random_states_function(
            states_spec=self.states_spec, actions_spec=self.actions_spec
        )
        self.is_valid_actions = self.__class__.is_valid_actions_function(
            actions_spec=self.actions_spec
        )

    def states(self):
        return self.states_spec

    def actions(self):
        return self.actions_spec

    def max_episode_timesteps(self):
        return self.timestep_range[1] + 1

    @classmethod
    def random_states_function(cls, states_spec, actions_spec=None):
        if actions_spec is None:
            if util.is_atomic_values_spec(values_spec=states_spec):
                return (lambda: cls.random_state_function(state_spec=states_spec)())
            else:
                return (lambda: {
                    name: cls.random_state_function(state_spec=state_spec)()
                    for name, state_spec in states_spec.items()
                })

        elif util.is_atomic_values_spec(values_spec=states_spec):
            if util.is_atomic_values_spec(values_spec=actions_spec):

                def fn():
                    random_states = cls.random_state_function(state_spec=states_spec)()
                    if actions_spec['type'] == 'int':
                        if not isinstance(random_states, dict):
                            random_states = dict(state=random_states)
                        mask = cls.random_mask(action_spec=actions_spec)
                        random_states['action_mask'] = mask
                    return random_states

            else:

                def fn():
                    random_states = cls.random_state_function(state_spec=states_spec)()
                    for name, action_spec in actions_spec.items():
                        if action_spec['type'] == 'int':
                            if not isinstance(random_states, dict):
                                random_states = dict(state=random_states)
                            mask = cls.random_mask(action_spec=action_spec)
                            random_states[name + '_mask'] = mask
                    return random_states

        else:
            if util.is_atomic_values_spec(values_spec=actions_spec):

                def fn():
                    random_states = {
                        name: cls.random_state_function(state_spec=state_spec)()
                        for name, state_spec in states_spec.items()
                    }
                    if actions_spec['type'] == 'int':
                        mask = cls.random_mask(action_spec=actions_spec)
                        random_states['action_mask'] = mask
                    return random_states

            else:

                def fn():
                    random_states = {
                        name: cls.random_state_function(state_spec=state_spec)()
                        for name, state_spec in states_spec.items()
                    }
                    for name, action_spec in actions_spec.items():
                        if action_spec['type'] == 'int':
                            mask = cls.random_mask(action_spec=action_spec)
                            random_states[name + '_mask'] = mask
                    return random_states

        return fn

    @classmethod
    def random_state_function(cls, state_spec):
        shape = state_spec['shape']
        dtype = state_spec.get('type', 'float')

        if dtype == 'bool':
            return (lambda: np.random.random_sample(size=shape) >= 0.5)

        elif dtype == 'int':
            num_values = state_spec['num_values']
            return (lambda: np.random.randint(low=0, high=num_values, size=shape))

        elif dtype == 'float':
            if 'min_value' in state_spec:
                min_value = state_spec['min_value']
                max_value = state_spec['max_value']
                return (lambda: (
                    min_value + (max_value - min_value) * np.random.random_sample(size=shape)
                ))

            else:
                return (lambda: np.random.standard_normal(size=shape))

    @classmethod
    def random_mask(cls, action_spec):
        shape = action_spec['shape'] + (action_spec['num_values'],)
        mask = np.random.random_sample(size=shape)
        min_mask = np.amin(mask, -1, keepdims=True)
        max_mask = np.amax(mask, -1, keepdims=True)
        threshold = np.random.random_sample(size=shape)
        mask = mask < min_mask + threshold * (max_mask - min_mask)
        assert mask.any(-1).all() and not mask.all(-1).any()
        return mask

    @classmethod
    def is_valid_actions_function(cls, actions_spec):
        if util.is_atomic_values_spec(values_spec=actions_spec):
            return (lambda actions, states:
                cls.is_valid_action_function(action_spec=actions_spec)(actions, 'action', states)
            )

        else:
            return (lambda actions, states: all(
                cls.is_valid_action_function(action_spec=action_spec)(
                    action=actions[name], name=name, states=states
                ) for name, action_spec in actions_spec.items()
            ))

    @classmethod
    def is_valid_action_function(cls, action_spec):
        dtype = action_spec['type']
        shape = action_spec.get('shape', ())

        if dtype == 'bool':
            return (lambda action, name, states: (
                (isinstance(action, util.np_dtype('bool')) and shape == ()) or
                (
                    isinstance(action, np.ndarray) and
                    action.dtype == util.np_dtype('bool') and action.shape == shape
                )
            ))

        elif dtype == 'int':
            num_values = action_spec['num_values']
            return (lambda action, name, states: (
                (
                    (isinstance(action, util.np_dtype('int')) and shape == ()) or
                    (
                        isinstance(action, np.ndarray) and
                        action.dtype == util.np_dtype('int') and action.shape == shape
                    )
                ) and (0 <= action).all() and (action < num_values).all() and
                np.take_along_axis(
                    states[name + '_mask'], indices=np.expand_dims(action, axis=-1), axis=-1
                ).all()
            ))

        elif dtype == 'float':
            if 'min_value' in action_spec:
                min_value = action_spec['min_value']
                max_value = action_spec['max_value']
                return (lambda action, name, states: (
                    (
                        (isinstance(action, util.np_dtype('float')) and shape == ()) or
                        (
                            isinstance(action, np.ndarray) and
                            action.dtype == util.np_dtype('float') and action.shape == shape
                        )
                    ) and (min_value <= action).all() and (action <= max_value).all()
                ))

            else:
                return (lambda action, name, states: (
                    (isinstance(action, util.np_dtype('float')) and shape == ()) or
                    (
                        isinstance(action, np.ndarray) and
                        action.dtype == util.np_dtype('float') and action.shape == shape
                    )
                ))

    def reset(self):
        self.num_timesteps = randint(*self.timestep_range)
        self.timestep = 0
        self.states = self.random_states()
        return self.states

    def execute(self, actions):
        if not self.is_valid_actions(actions, self.states):
            raise TensorforceError.value(name='actions', value=actions)
        if self.timestep > self.num_timesteps:
            raise TensorforceError.unexpected()

        self.timestep += 1
        self.states = self.random_states()
        terminal = self.timestep == self.num_timesteps
        reward = -1.0 + 2.0 * random()

        return self.states, terminal, reward
