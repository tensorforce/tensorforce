# Copyright 2018 TensorForce Team. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from random import randint, random

import numpy as np

from tensorforce import util, TensorForceError
from tensorforce.environments import Environment


class UnittestEnvironment(Environment):
    """
    Unit-test environment.
    """

    def __init__(self, states, actions):
        """
        Initializes a mock environment which is used for the unit-tests.

        Args:
            states: the state specification for the unit-test
            actions: the action specification for the unit-test
        """
        if not util.is_valid_state_spec(state_spec=states):
            raise TensorForceError("Invalid state specification.")
        if not util.is_valid_action_spec(action_spec=actions):
            raise TensorForceError("Invalid action specification.")

        self.state_spec = states
        self.action_spec = actions
        self.random_state = self.__class__.random_state_function(state_spec=states)
        self.is_valid_action = self.__class__.is_valid_action_function(action_spec=actions)

    @property
    def states(self):
        return self.state_spec

    @property
    def actions(self):
        return self.action_spec

    @classmethod
    def random_state_function(cls, state_spec):
        if util.is_single_state(state_spec=state_spec):
            return cls.random_state_component_function(state_spec=state_spec)

        else:
            return (lambda: {
                name: cls.random_state_component_function(state_spec=state_spec[name])()
                for name in sorted(state_spec)
            })

    @classmethod
    def random_state_component_function(cls, state_spec):
        shape = state_spec['shape']
        dtype = state_spec.get('type', 'float')

        if dtype == 'bool':
            return (lambda: np.random.random_sample(size=shape) >= 0.5)

        elif dtype == 'int':
            num_actions = state_spec['num_states']
            return (lambda: np.random.randint(low=0, high=num_actions, size=shape))

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
    def is_valid_action_function(cls, action_spec):
        if util.is_single_action(action_spec=action_spec):
            return cls.is_valid_action_component_function(action_spec=action_spec)

        else:
            return (lambda action: all(
                cls.is_valid_action_component_function(action_spec=action_spec[name])(
                    action=action[name]
                ) for name in sorted(action_spec)
            ))

    @classmethod
    def is_valid_action_component_function(cls, action_spec):
        dtype = action_spec['type']
        shape = action_spec.get('shape', ())

        if dtype == 'bool':
            return (lambda action: (
                isinstance(action, util.np_dtype('bool')) or \
                (isinstance(action, np.ndarray) and action.dtype == util.np_dtype('bool'))
            ))

        elif dtype == 'int':
            num_actions = action_spec['num_actions']
            return (lambda action: (
                (
                    isinstance(action, util.np_dtype('int')) or \
                    (isinstance(action, np.ndarray) and action.dtype == util.np_dtype('int'))
                ) and (0 <= action).all() and (action < num_actions).all()
            ))

        elif dtype == 'float':
            if 'min_value' in action_spec:
                min_value = action_spec['min_value']
                max_value = action_spec['max_value']
                return (lambda action: (
                    (
                        isinstance(action, util.np_dtype('float')) or
                        (isinstance(action, np.ndarray) and action.dtype == util.np_dtype('float'))
                    ) and (min_value <= action).all() and (action <= max_value).all()
                ))

            else:
                return (lambda action: (
                    isinstance(action, util.np_dtype('float')) or \
                    (isinstance(action, np.ndarray) and action.dtype == util.np_dtype('float'))
                ))

    def reset(self):
        self.num_timesteps = randint(1, 10)
        self.timestep = 0
        state = self.random_state()
        return state

    def execute(self, action):
        if not self.is_valid_action(action=action):
            print(action)
            raise TensorForceError("Invalid action.")

        state = self.random_state()
        terminal = self.timestep < self.num_timesteps
        reward = -1.0 + 2.0 * random()
        return state, terminal, reward
