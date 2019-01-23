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

from threading import Thread

from tensorforce import TensorforceError


class Environment(object):
    """
    Environment base class.
    """

    def __init__(self):
        self.observation = None
        self.thread = None

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
        if self.thread is not None:
            self.thread.join()
        self.observation = None
        self.thread = None

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        raise NotImplementedError
        # if self.observation is not None or self.thread is not None:
        #     raise TensorforceError(message="Invalid execute.")
        # self.start_reset()
        # self.thread.join()
        # states, _, _ = self.observe()
        # if self.observation is not None:
        #     raise TensorforceError(message="Invalid start_reset/observe implementation.")
        # return states

    def execute(self, actions):
        """
        Executes action, observes next state(s) and reward.

        Args:
            actions: Actions to execute.

        Returns:
            Tuple of (next state, bool indicating terminal, reward)
        """
        raise NotImplementedError
        # if self.observation is not None or self.thread is not None:
        #     raise TensorforceError(message="Invalid execute.")
        # self.start_execute(actions=actions)
        # self.thread.join()
        # observation = self.observe()
        # if self.observation is not None:
        #     raise TensorforceError(message="Invalid start_execute/observe implementation.")
        # return observation

    def start_reset(self):
        if self.thread is not None:
            raise TensorforceError(message="Invalid start_reset.")
        self.thread = Thread(target=self.finish_reset)
        self.thread.start()

    def finish_reset(self):
        self.observation = (self.reset(), None, None)
        self.thread = None

    def start_execute(self, actions):
        if self.observation is not None or self.thread is not None:
            raise TensorforceError(message="Invalid start_execute.")
        self.thread = Thread(target=self.finish_execute, kwargs=dict(actions=actions))
        self.thread.start()

    def finish_execute(self, actions):
        self.observation = self.execute(actions=actions)
        self.thread = None

    def retrieve_execute(self):
        if self.thread is not None:
            return None
        else:
            if self.observation is None:
                raise TensorforceError(message="Invalid retrieve_execute.")
            observation = self.observation
            self.observation = None
            return observation
