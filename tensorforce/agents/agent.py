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

"""Agent base class."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
from six.moves import xrange
from random import random
import numpy as np

from tensorforce import util, TensorForceError
from tensorforce.core.preprocessing import Preprocessing
from tensorforce.core.explorations import Exploration


class Agent(object):
    """
    Basic Reinforcement learning agent. An agent encapsulates execution logic
    of a particular reinforcement learning algorithm and defines the external interface
    to the environment.

    The agent hence acts an intermediate layer between environment
    and backend execution (value function or policy updates).

    Each agent requires the following configuration parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    The configuration is passed to the [Model](#Model) and should thus include its configuration parameters, too.

    Examples:

        One state, one action, two preprecessors, epsilon exploration.

        ```python
        agent = Agent(Configuration(dict(
            states=dict(shape=(10,), type='float'),
            actions=dict(continuous=False, num_actions=6),
            preprocessing=[dict(type="sequence", args=[4]), dict=(type="max", args=[2])],
            exploration=...,
            # ... model configuration parameters
        )))
        ```

        Two states, two actions:

        ```python

        agent = Agent(Configuration(dict(
            states=dict(
                state1=dict(shape=(10,), type='float'),
                state2=dict(shape=(40,20), type='int')
            ),
            actions=dict(
                action1=dict(continuous=True),
                action2=dict(continuous=False, num_actions=6)
            ),
            preprocessing=dict(
                state1=[dict(type="sequence", args=[4]), dict=(type="max", args=[2])],
                state2=None
            ),
            exploration=dict(
                action1=...,
                action2=...
            ),
            # ... model configuration parameters
        )))
        ```
    """

    name = None
    model = None
    default_config = dict(
        preprocessing=None,
        exploration=None,
        log_level='info'
    )

    def __init__(self, config, model=None):
        """Initializes the reinforcement learning agent.

        Args:
            config (Configuration): configuration object containing at least `states`, `actions`, `preprocessing` and
                'exploration`.
            model (Model): optional model instance. If not supplied, a new model is created.

        """
        assert self.__class__.name is not None and self.__class__.model is not None
        config.default(Agent.default_config)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(util.log_levels[config.log_level])

        # states config and preprocessing
        self.preprocessing = dict()
        if 'shape' in config.states:
            # only one state
            config.states = dict(state=config.states)
            self.unique_state = True
            if config.preprocessing is not None:
                config.preprocessing = dict(state=config.preprocessing)
        else:
            self.unique_state = False
        for name, state in config.states:
            state.default(dict(type='float'))
            if isinstance(state.shape, int):
                state.shape = (state.shape,)
            if config.preprocessing is not None and name in config.preprocessing:
                preprocessing = Preprocessing.from_config(config=config.preprocessing[name])
                self.preprocessing[name] = preprocessing
                state.shape = preprocessing.processed_shape(shape=state.shape)

        # actions config and exploration
        self.exploration = dict()
        if 'continuous' in config.actions:
            # only one action
            config.actions = dict(action=config.actions)
            if config.exploration is not None:
                config.exploration = dict(action=config.exploration)
            self.unique_action = True
        else:
            self.unique_action = False
        for name, action in config.actions:
            if action.continuous:
                action.default(dict(shape=(), min_value=None, max_value=None))
            else:
                action.default(dict(shape=()))
            if isinstance(action.shape, int):
                action.shape = (action.shape,)
            if config.exploration is not None and name in config.exploration:
                self.exploration[name] = Exploration.from_config(config=config.exploration[name])

        self.states_config = config.states
        self.actions_config = config.actions

        if model is None:
            self.model = self.__class__.model(config)
        else:
            if not isinstance(model, self.__class__.model):
                raise TensorForceError("Supplied model class `{}` does not match expected agent model class `{}`".format(
                    type(model).__name__, self.__class__.model.__name__
                ))
            self.model = model

        not_accessed = config.not_accessed()
        if not_accessed:
            self.logger.warning("Configuration values not accessed: {}".format(', '.join(not_accessed)))

        self.episode = -1
        self.timestep = 0
        self.reset()

    def __str__(self):
        return str(self.__class__.name)

    def reset(self):
        """Reset agent after episode. Increments internal episode count, internal states and preprocessors.

        Returns:
            void

        """
        self.episode += 1
        self.current_internal = self.next_internal = self.model.reset()
        for preprocessing in self.preprocessing.values():
            preprocessing.reset()

    def act(self, state, deterministic=False):
        """Return action(s) for given state(s). First, the states are preprocessed using the given preprocessing
        configuration. Then, the states are passed to the model to calculate the desired action(s) to execute.

        After obtaining the actions, exploration might be added by the agent, depending on the exploration
        configuration.

        Args:
            state: One state (usually a value tuple) or dict of states if multiple states are expected.
            deterministic: If true, no exploration and sampling is applied.

        Returns:
            Scalar value of the action or dict of multiple actions the agent wants to execute.

        """
        self.timestep += 1
        self.current_internal = self.next_internal

        if self.unique_state:
            self.current_state = dict(state=state)
        else:
            self.current_state = state

        # preprocessing
        for name, preprocessing in self.preprocessing.items():
            self.current_state[name] = preprocessing.process(state=self.current_state[name])

        # model action
        self.current_action, self.next_internal = self.model.get_action(state=self.current_state, internal=self.current_internal, deterministic=deterministic)

        # exploration
        if not deterministic:
            for name, exploration in self.exploration.items():
                if self.actions_config[name].continuous:
                    explore = (lambda: exploration(episode=self.episode, timestep=self.timestep))
                    shape = self.actions_config[name].shape
                    exploration = np.array([explore() for _ in xrange(util.prod(shape))])
                    self.current_action[name] += np.reshape(exploration, shape)
                else:
                    if random() < exploration(episode=self.episode, timestep=self.timestep):
                        shape = self.actions_config[name].shape
                        num_actions = self.actions_config[name].num_actions
                        self.current_action[name] = np.random.randint(low=num_actions, size=shape)

        if self.unique_action:
            return self.current_action['action']
        else:
            return self.current_action

    def observe(self, reward, terminal):
        """Observe experience from the environment to learn from.

        Args:
            reward: scalar reward that resulted from executing the action.
            terminal: boolean indicating if the episode terminated after the observation.

        Returns:
            void

        """
        raise NotImplementedError

    def observe_episode_reward(self, episode_reward):
        self.model.write_episode_reward_summary(episode_reward)

    def last_observation(self):
        return dict(
            state=self.current_state,
            action=self.current_action,
            reward=self.current_reward,
            terminal=self.current_terminal,
            internal=self.current_internal
        )

    def load_model(self, path):
        self.model.load_model(path)

    def save_model(self, path):
        self.model.save_model(path)
