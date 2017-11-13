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

from six.moves import xrange
from random import random
import numpy as np

from tensorforce import util, TensorForceError
from tensorforce.core.preprocessing import Preprocessing
from tensorforce.core.explorations import Exploration
import tensorforce.agents


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

    def __init__(
        self,
        states_spec,
        actions_spec,
        preprocessing,
        exploration,
        reward_preprocessing,
        batched_observe
    ):
        """
        Initializes the reinforcement learning agent.

        Args:
            states_spec:
            actions_spec:
            preprocessing:
            exploration:
            reward_preprocessing:
            batched_observe:
        """

        # States config and preprocessing
        self.preprocessing = dict()

        self.unique_state = False
        if 'shape' in states_spec.keys():
            self.unique_state = True
            states_spec = dict(state=states_spec)
            preprocessing = dict(state=preprocessing)

        self.states_spec = states_spec

        for name, state in self.states_spec.items():
            # Convert int to unary tuple
            if isinstance(state['shape'], int):
                state['shape'] = (state['shape'],)

            # Set default type to float
            if 'type' not in state.keys():
                state['type'] = 'float'

            if preprocessing is not None and name in preprocessing.keys() and preprocessing[name]:
                state_preprocessing = Preprocessing.from_spec(preprocessing[name])
                self.preprocessing[name] = state_preprocessing
                state['shape'] = state_preprocessing.processed_shape(shape=state['shape'])

        # Actions config and exploration
        self.exploration = dict()

        self.unique_action = False
        if 'type' in actions_spec.keys():
            self.unique_action = True
            actions_spec = dict(action=actions_spec)
            exploration = dict(action=exploration)

        self.actions_spec = actions_spec

        for name, action in self.actions_spec.items():
            # Check requried values
            if action['type'] == 'int':
                if 'num_actions' not in action.keys():
                    raise TensorForceError("Action requires value 'num_actions' set!")
            elif action['type'] == 'float':
                if ('min_value' in action.keys()) != ('max_value' in action):
                    raise TensorForceError("Action requires both values 'min_value' and 'max_value' set!")

            # Set default shape to empty tuple
            if 'shape' not in action.keys():
                action['shape'] = ()

            # Convert int to unary tuple
            if isinstance(action['shape'], int):
                action['shape'] = (action['shape'],)

            # Set exploration
            if exploration is not None and name in exploration.keys() and exploration[name]:
                self.exploration[name] = Exploration.from_spec(exploration[name])

        # reward preprocessing config
        if reward_preprocessing is None:
            self.reward_preprocessing = None
        else:
            self.reward_preprocessing = Preprocessing.from_spec(reward_preprocessing)

        self.model = self.initialize_model(
            states_spec=self.states_spec,
            actions_spec=self.actions_spec,
        )

        # Batched observe for better performance with Python.
        self.batched_observe = batched_observe
        if self.batched_observe is not None:
            self.observe_terminal = list()
            self.observe_reward = list()

        self.reset()

    def __str__(self):
        return str(self.__class__.__name__)

    def close(self):
        self.model.close()

    def initialize_model(self, states_spec, actions_spec):
        raise NotImplementedError

    def reset(self):
        """
        Reset the agent to its initial state on episode start. Updates internal episode and  
        timestep counter, internal states,  and resets preprocessors.
        """
        self.episode, self.timestep, self.next_internals = self.model.reset()
        self.current_internals = self.next_internals

        for preprocessing in self.preprocessing.values():
            preprocessing.reset()

    def act(self, states, deterministic=False):
        """
        Return action(s) for given state(s). First, the states are preprocessed using the given preprocessing
        configuration. Then, the states are passed to the model to calculate the desired action(s) to execute.

        After obtaining the actions, exploration might be added by the agent, depending on the exploration
        configuration.

        Args:
            states: One state (usually a value tuple) or dict of states if multiple states are expected.
            deterministic: If true, no exploration and sampling is applied.
        Returns:
            Scalar value of the action or dict of multiple actions the agent wants to execute.

        """

        self.current_internals = self.next_internals

        if self.unique_state:
            self.current_states = dict(state=np.asarray(states))
        else:
            self.current_states = {name: np.asarray(state) for name, state in states.items()}

        # Preprocessing
        for name, preprocessing in self.preprocessing.items():
            self.current_states[name] = preprocessing.process(state=self.current_states[name])

        # Retrieve action
        self.current_actions, self.next_internals, self.timestep = self.model.act(
            states=self.current_states,
            internals=self.current_internals,
            deterministic=deterministic
        )

        # Exploration
        if not deterministic:
            for name, exploration in self.exploration.items():

                if self.actions_spec[name]['type'] == 'bool':
                    if random() < exploration(episode=self.episode, timestep=self.timestep):
                        shape = self.actions_spec[name]['shape']
                        self.current_actions[name] = (np.random.random_sample(size=shape) < 0.5)

                elif self.actions_spec[name]['type'] == 'int':
                    if random() < exploration(episode=self.episode, timestep=self.timestep):
                        shape = self.actions_spec[name]['shape']
                        num_actions = self.actions_spec[name]['num_actions']
                        self.current_actions[name] = np.random.randint(low=num_actions, size=shape)

                elif self.actions_spec[name]['type'] == 'float':
                    explore = (lambda: exploration(episode=self.episode, timestep=self.timestep))
                    shape = self.actions_spec[name]['shape']
                    exploration = np.array([explore() for _ in xrange(util.prod(shape))])

                    if 'min_value' in self.actions_spec[name]:
                        exploration = np.clip(
                            a=exploration,
                            a_min=self.actions_spec[name]['min_value'],
                            a_max=self.actions_spec[name]['max_value']
                        )

                    self.current_actions[name] += np.reshape(exploration, shape)

        if self.unique_action:
            return self.current_actions['action']
        else:
            return self.current_actions

    def observe(self, terminal, reward):
        """
        Observe experience from the environment to learn from. Optionally preprocesses rewards
        Child classes should call super to get the processed reward
        EX: terminal, reward = super()...

        Args:
            terminal: boolean indicating if the episode terminated after the observation.
            reward: scalar reward that resulted from executing the action.
        """
        self.current_terminal = terminal
        if self.reward_preprocessing is None:
            self.current_reward = reward
        else:
            self.current_reward = self.reward_preprocessing.process(reward)

        if self.batched_observe > 0:
            # Batched observe for better performance with Python.
            self.observe_terminal.append(self.current_terminal)
            self.observe_reward.append(self.current_reward)

            if self.current_terminal or len(self.observe_terminal) >= self.batched_observe:
                self.episode = self.model.observe(
                    terminal=self.observe_terminal,
                    reward=self.observe_reward
                )
                self.observe_terminal = list()
                self.observe_reward = list()

        else:
            self.episode = self.model.observe(
                terminal=self.current_terminal,
                reward=self.current_reward
            )

    def should_stop(self):
        return self.model.monitored_session.should_stop()

    def last_observation(self):
        return dict(
            states=self.current_states,
            internals=self.current_internals,
            actions=self.current_actions,
            terminal=self.current_terminal,
            reward=self.current_reward
        )

    def save_model(self, directory=None, append_timestep=True):
        """
        Save TensorFlow model. If no checkpoint directory is given, the model's default saver  
        directory is used. Optionally appends current timestep to prevent overwriting previous  
        checkpoint files. Turn off to be able to load model from the same given path argument as  
        given here.

        Args:
            directory: Optional checkpoint directory.
            use_global_step:  Appends the current timestep to the checkpoint file if true.
            If this is set to True, the load path must include the checkpoint timestep suffix.  
            For example, if stored to models/ and set to true, the exported file will be of the  
            form models/model.ckpt-X where X is the last timestep saved. The load path must  
            precisely match this file name. If this option is turned off, the checkpoint will  
            always overwrite the file specified in path and the model can always be loaded under  
            this path.

        Returns:
            Checkpoint path were the model was saved.
        """
        return self.model.save(directory=directory, append_timestep=append_timestep)

    def restore_model(self, directory=None, file=None):
        """
        Restore TensorFlow model. If no checkpoint file is given, the latest checkpoint is  
        restored. If no checkpoint directory is given, the model's default saver directory is  
        used (unless file specifies the entire path).

        Args:
            directory: Optional checkpoint directory.
            file: Optional checkpoint file, or path if directory not given.
        """
        self.model.restore(directory=directory, file=file)

    @staticmethod
    def from_spec(spec, kwargs):
        """
        Creates an agent from a specification dict.
        """
        agent = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.agents.agents,
            kwargs=kwargs
        )
        assert isinstance(agent, Agent)
        return agent
