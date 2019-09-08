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

import importlib
import json
import os
from threading import Thread

from tensorforce import TensorforceError, util
import tensorforce.environments


class Environment(object):
    """
    Tensorforce environment interface.
    """

    @staticmethod
    def create(environment, **kwargs):
        """
        Creates an environment from a specification.

        Args:
            environment (specification): JSON file, specification key, configuration dictionary,
                library module, or `Environment` subclass
                (<span style="color:#C00000"><b>required</b></span>).
            kwargs: Additional arguments.
        """
        if isinstance(environment, Environment):
            # TODO: asserts???????
            return environment

        elif isinstance(environment, dict):
            # Dictionary specification
            util.deep_disjoint_update(target=kwargs, source=environment)
            environment = kwargs.pop('environment', kwargs.pop('type', 'default'))
            assert environment is not None

            return Environment.create(environment=environment, **kwargs)

        elif isinstance(environment, str):
            if os.path.isfile(environment):
                # JSON file specification
                with open(environment, 'r') as fp:
                    environment = json.load(fp=fp)

                util.deep_disjoint_update(target=kwargs, source=environment)
                environment = kwargs.pop('environment', kwargs.pop('type', 'default'))
                assert environment is not None

                return Environment.create(environment=environment, **kwargs)

            elif '.' in environment:
                # Library specification
                library_name, module_name = environment.rsplit('.', 1)
                library = importlib.import_module(name=library_name)
                environment = getattr(library, module_name)

                environment = environment(**kwargs)
                assert isinstance(environment, Environment)

                return environment

            else:
                # Keyword specification
                environment = tensorforce.environments.environments[environment](**kwargs)
                assert isinstance(environment, Environment)

                return environment

        else:
            assert False

    def __init__(self):
        # first two arguments, if applicable: level, visualize=False
        self.observation = None
        self.thread = None

    def __str__(self):
        return self.__class__.__name__

    def states(self):
        """
        Returns the state space specification.

        Returns:
            specification: Arbitrarily nested dictionary of state descriptions with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; state data type
            (<span style="color:#00C000"><b>default</b></span>: "float").</li>
            <li><b>shape</b> (<i>int | iter[int]</i>) &ndash; state shape
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>num_states</b> (<i>int > 0</i>) &ndash; number of discrete state values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum state value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        """
        raise NotImplementedError

    def actions(self):
        """
        Returns the action space specification.

        Returns:
            specification: Arbitrarily nested dictionary of action descriptions with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; action data type
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>shape</b> (<i>int > 0 | iter[int > 0]</i>) &ndash; action shape
            (<span style="color:#00C000"><b>default</b></span>: scalar).</li>
            <li><b>num_actions</b> (<i>int > 0</i>) &ndash; number of discrete action values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum action value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        """
        raise NotImplementedError

    def max_episode_timesteps(self):
        """
        Returns the maximum number of timesteps per episode.

        Returns:
            int: Maximum number of timesteps per episode.
        """
        return None

    def close(self):
        """
        Closes the environment.
        """
        if self.thread is not None:
            self.thread.join()
        self.observation = None
        self.thread = None

    def reset(self):
        """
        Resets the environment to start a new episode.

        Returns:
            dict[state]: Dictionary containing initial state(s) and auxiliary information.
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
        Executes the given action(s) and advances the environment by one step.

        Args:
            actions (dict[action]): Dictionary containing action(s) to be executed
                (<span style="color:#C00000"><b>required</b></span>).

        Returns:
            ((dict[state], bool | 0 | 1 | 2, float)): Dictionary containing next state(s), whether
            a terminal state is reached or 2 if the episode was aborted, and observed reward.
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
