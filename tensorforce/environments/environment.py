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

from tensorforce import TensorforceError


class Environment(object):
    """
    Environment base class.
    """

    @property
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are 
        available simultaneously.

        Returns:
            States specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (default: 'float').
                - shape: integer, or list/tuple of integers (required).
        """
        raise NotImplementedError

    @property
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are 
        available simultaneously.

        Returns:
            actions (spec, or dict of specs): Actions specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (required).
                - shape: integer, or list/tuple of integers (default: []).
                - num_actions: integer (required if type == 'int').
                - min_value and max_value: float (optional if type == 'float', default: none).
        """
        raise NotImplementedError

    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments (e.g. ALE or some gym Envs) don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """
        return None

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        pass

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        if not hasattr(self, '_observation'):
            self._observation = None
        if self._observation is not None:
            raise TensorforceError(message="Invalid execute.")
        self.just_reset()
        states, _, _ = self.observe()
        if self._observation is not None:
            raise TensorforceError(message="Invalid just_reset/observe implementation.")
        return states

    def execute(self, actions):
        """
        Executes action, observes next state(s) and reward.

        Args:
            actions: Actions to execute.

        Returns:
            Tuple of (next state, bool indicating terminal, reward)
        """
        if self._observation is not None:
            raise TensorforceError(message="Invalid execute.")
        self.just_execute(actions=actions)
        observation = self.observe()
        if self._observation is not None:
            raise TensorforceError(message="Invalid just_execute/observe implementation.")
        return observation

    def just_reset(self):
        if not hasattr(self, '_observation'):
            self._observation = None
        if self._observation is not None:
            raise TensorforceError(message="Invalid execute.")
        self._observation = (self.reset(), None, None)

    def just_execute(self, actions):
        if self._observation is not None:
            raise TensorforceError(message="Invalid just)execute.")
        self._observation = self.execute(actions=actions)

    def observe(self):
        if self._observation is None:
            raise TensorforceError(message="Invalid observe.")
        observation = self._observation
        self._observation = None
        return observation
