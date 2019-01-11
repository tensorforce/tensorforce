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

import retro
import gym
import numpy as np

from tensorforce import TensorForceError
from tensorforce.environments import Environment


class OpenAIRetro(Environment):
    """
    Bindings for OpenAIRetro environments. Works similar to OpenAI Gym.
    """

    def __init__(self, game, state=None):
        if state is None:
            self.env = retro.make(game)
        else:
            self.env = retro.make(game, state=state)

    def __str__(self):
        raise NotImplementedError

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        self.env = None

    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments (e.g. ALE or some gym Envs) don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """
        return self.seed(seed)[1]

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        return self.env.reset()

    def execute(self, action):
        """
        Executes action, observes next state and reward.

        Args:
            actions: Actions to execute.

        Returns:
            Tuple of (next state, bool indicating terminal, reward)
        """
        next_state, rew, done, _ = self.env.step(action)
        return next_state, rew, done

    def states(self):
        return OpenAIRetro.state_from_space(space=self.env.observation_space)

    @staticmethod
    def state_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(shape=(), type='int')
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(shape=space.n, type='int')
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return dict(shape=space.num_discrete_space, type='int')
        elif isinstance(space, gym.spaces.Box):
            return dict(shape=tuple(space.shape), type='float')
        elif isinstance(space, gym.spaces.Tuple):
            states = dict()
            n = 0
            for space in space.spaces:
                state = OpenAIRetro.state_from_space(space=space)
                if 'type' in state:
                    states['state{}'.format(n)] = state
                    n += 1
                else:
                    for state in state.values():
                        states['state{}'.format(n)] = state
                        n += 1
            return states

        elif isinstance(space, gym.spaces.Dict):
            states = dict()
            for space_name,space in space.spaces.items():
                state = OpenAIRetro.state_from_space(space=space)
                states[space_name] = state
            return states
        else:
            raise TensorForceError('Unknown Gym space.')

    def actions(self):
        return OpenAIRetro.action_from_space(space=self.env.action_space)

    @staticmethod
    def action_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(type='int', num_actions=space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(type='bool', shape=space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            num_discrete_space = len(space.nvec)
            if (space.nvec == space.nvec[0]).all():
                return dict(type='int', num_actions=space.nvec[0], shape=num_discrete_space)
            else:
                actions = dict()
                for n in range(num_discrete_space):
                    actions['action{}'.format(n)] = dict(type='int', num_actions=space.nvec[n])
                return actions
        elif isinstance(space, gym.spaces.Box):
            if (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(type='float', shape=space.low.shape,
                            min_value=np.float32(space.low[0]),
                            max_value=np.float32(space.high[0]))
            else:
                actions = dict()
                low = space.low.flatten()
                high = space.high.flatten()
                for n in range(low.shape[0]):
                    actions['action{}'.format(n)] = dict(type='float', min_value=low[n], max_value=high[n])
                return actions
        elif isinstance(space, gym.spaces.Tuple):
            actions = dict()
            n = 0
            for space in space.spaces:
                action = OpenAIRetro.action_from_space(space=space)
                if 'type' in action:
                    actions['action{}'.format(n)] = action
                    n += 1
                else:
                    for action in action.values():
                        actions['action{}'.format(n)] = action
                        n += 1
            return actions
        elif isinstance(space, gym.spaces.Dict):
            actions = dict()
            for space_name,space in space.spaces.items():
                action = OpenAIRetro.action_from_space(space=space)
                actions[space_name] = action
            return actions

        else:
            raise TensorForceError('Unknown Gym space.')
