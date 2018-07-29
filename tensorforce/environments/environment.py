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


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorforce.environments
import tensorforce.util


class Environment(object):
    """
    Base environment class.
    """

    def __str__(self):
        raise NotImplementedError

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        pass

    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments (e.g. ALE or some gym Envs) don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """
        return None

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        raise NotImplementedError

    def execute(self, action):
        """
        Executes action, observes next state(s) and reward.

        Args:
            actions: Actions to execute.

        Returns:
            Tuple of (next state, bool indicating terminal, reward)
        """
        raise NotImplementedError

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

    @staticmethod
    def from_spec(spec, kwargs):
        """
        Creates an environment from a specification dict.
        """
        env = tensorforce.util.get_object(
            obj=spec,
            predefined_objects=tensorforce.environments.environments,
            kwargs=kwargs
        )
        assert isinstance(env, Environment)
        return env
