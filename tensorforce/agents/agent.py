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

from copy import deepcopy

import numpy as np
import inspect

from tensorforce import util, TensorForceError
import tensorforce.agents
from tensorforce.meta_parameter_recorder import MetaParameterRecorder


class Agent(object):
    """
    Basic Reinforcement learning agent. An agent encapsulates execution logic
    of a particular reinforcement learning algorithm and defines the external interface
    to the environment.

    The agent hence acts as an intermediate layer between environment
    and backend execution (value function or policy updates).

    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe=1000,
        scope='base_agent'
    ):
        """
        Initializes the reinforcement learning agent.

        Args:
            states_spec (dict): Dict containing at least one state-component definition. In the case of a single state
                space component, the keys `shape` and `type` are necessary (e.g. a 3D float-box with shape [3,3,3]).
                For multiple state components, pass a dict of dicts where each component is a dict itself with a unique
                name as its key (e.g. {cam: {shape: [84,84], type=int}, health: {shape=(), type=float}}).
            actions_spec (dict): Dict containing at least one action-component definition.
                Action components have types and either `num_actions` for discrete actions or a `shape`
                for continuous actions.
                Consult documentation and tests for more.
            batched_observe (int): How many calls to `observe` are batched into one tensorflow session run.
                Values of 0 or 1 indicate no batching being used and every call to `observe` triggers a tensorflow
                session invocation to update rewards in the graph, which will lower the throughput.
            scope: TensorFlow scope, defaults to agent name (e.g. `dqn`).
        """

        # process state space
        self.states_spec, self.unique_state = self.process_state_spec(states_spec)

        # Actions config and exploration
        self.exploration = dict()
        self.actions_spec, self.unique_action = self.process_action_spec(actions_spec)

        # Batched observe for better performance with Python.
        self.batched_observe = batched_observe
        if self.batched_observe is not None:
            self.observe_terminal = list()
            self.observe_reward = list()

        self.scope = scope

        # Init Model, this must follow the Summary Configuration section above to cary meta_param_recorder
        self.model = self.initialize_model()

        #  Define the properties used to store internal state of Agent.
        self.current_states = None
        self.current_actions = None
        self.current_internals = None
        self.next_internals = None
        self.current_terminal = None
        self.current_reward = None
        self.episode = None
        self.timestep = None

        self.reset()

    def __str__(self):
        return str(self.__class__.__name__)

    def close(self):
        self.model.close()

    def initialize_model(self):
        """
        Creates the model for the respective agent based on specifications given by user. This is a separate
        call after constructing the agent because the agent constructor has to perform a number of checks
        on the specs first, sometimes adjusting them e.g. by converting to a dict.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the agent to its initial state on episode start. Updates internal episode and  
        timestep counter, internal states,  and resets preprocessors.
        """
        self.episode, self.timestep, self.next_internals = self.model.reset()
        self.current_internals = self.next_internals

        # TODO have to call preprocessing reset in model
        # for preprocessing in self.preprocessing.values():
        #     preprocessing.reset()

    def act(self, states, deterministic=False):
        """
        Return action(s) for given state(s). States preprocessing and exploration are applied if  
        configured accordingly.

        Args:
            states (any): One state (usually a value tuple) or dict of states if multiple states are expected.
            deterministic (bool): If true, no exploration and sampling is applied.
        Returns:
            Scalar value of the action or dict of multiple actions the agent wants to execute.

        """
        self.current_internals = self.next_internals

        if self.unique_state:
            self.current_states = dict(state=np.asarray(states))
        else:
            self.current_states = {name: np.asarray(state) for name, state in states.items()}

        # Retrieve action
        self.current_actions, self.next_internals, self.timestep = self.model.act(
            states=self.current_states,
            internals=self.current_internals,
            deterministic=deterministic
        )

        if self.unique_action:
            return self.current_actions['action']
        else:
            return self.current_actions

    def observe(self, terminal, reward):
        """
        Observe experience from the environment to learn from. Optionally pre-processes rewards
        Child classes should call super to get the processed reward
        EX: terminal, reward = super()...

        Args:
            terminal (bool): boolean indicating if the episode terminated after the observation.
            reward (float): scalar reward that resulted from executing the action.
        """
        self.current_terminal = terminal
        self.current_reward = reward

        if self.batched_observe is not None and self.batched_observe > 1:
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
            directory (str): Optional checkpoint directory.
            append_timestep (bool):  Appends the current timestep to the checkpoint file if true.
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

    @staticmethod
    def process_state_spec(states_spec):
        unique_state = ('shape' in states_spec)

        # Leave incoming spec-dict intact.
        states_spec_copy = deepcopy(states_spec)
        if unique_state:
            states_spec_copy = dict(state=states_spec_copy)

        for name, state in states_spec_copy.items():
            # Convert int to unary tuple.
            if isinstance(state['shape'], int):
                state['shape'] = (state['shape'],)

            # Set default type to float.
            if 'type' not in state:
                state['type'] = 'float'
        return states_spec_copy, unique_state

    @staticmethod
    def process_action_spec(actions_spec):
        unique_action = ('type' in actions_spec)
        # Leave incoming spec-dict intact.
        actions_spec_copy = deepcopy(actions_spec)
        if unique_action:
            actions_spec_copy = dict(action=actions_spec_copy)

        for name, action in actions_spec_copy.items():
            # Set default type to int
            if 'type' not in action:
                action['type'] = 'int'

            # Check required values
            if action['type'] == 'int':
                if 'num_actions' not in action:
                    raise TensorForceError("Action requires value 'num_actions' set!")
            elif action['type'] == 'float':
                if ('min_value' in action) != ('max_value' in action):
                    raise TensorForceError("Action requires both values 'min_value' and 'max_value' set!")

            # Set default shape to empty tuple (single-int, discrete action space)
            if 'shape' not in action:
                action['shape'] = ()

            # Convert int to unary tuple
            if isinstance(action['shape'], int):
                action['shape'] = (action['shape'],)

        return actions_spec_copy, unique_action
