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
        batched_observe
    ):
        """
        Initializes the reinforcement learning agent.

        Args:
            states_spec: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions_spec: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
        """

        self.unique_state = ('shape' in states_spec)
        if self.unique_state:
            states_spec = dict(state=states_spec)

        self.states_spec = deepcopy(states_spec)
        for name, state in self.states_spec.items():
            # Convert int to unary tuple
            if isinstance(state['shape'], int):
                state['shape'] = (state['shape'],)

            # Set default type to float
            if 'type' not in state:
                state['type'] = 'float'

        # Actions config and exploration
        self.exploration = dict()
        self.unique_action = ('type' in actions_spec)
        if self.unique_action:
            actions_spec = dict(action=actions_spec)
        self.actions_spec = deepcopy(actions_spec)

        for name, action in self.actions_spec.items():
            # Check required values
            if action['type'] == 'int':
                if 'num_actions' not in action:
                    raise TensorForceError("Action requires value 'num_actions' set!")
            elif action['type'] == 'float':
                if ('min_value' in action) != ('max_value' in action):
                    raise TensorForceError("Action requires both values 'min_value' and 'max_value' set!")

            # Set default shape to empty tuple
            if 'shape' not in action:
                action['shape'] = ()

            # Convert int to unary tuple
            if isinstance(action['shape'], int):
                action['shape'] = (action['shape'],)

        # TensorFlow summaries & Configuration Meta Parameter Recorder options
        if self.summary_spec is None:
            self.summary_labels = set()
        else:
            self.summary_labels = set(self.summary_spec.get('labels', ()))
 
        self.meta_param_recorder = None
 
        #if 'configuration' in self.summary_labels or 'print_configuration' in self.summary_labels:
        if any(k in self.summary_labels for k in ['configuration','print_configuration']):
            self.meta_param_recorder = MetaParameterRecorder(inspect.currentframe())
            if 'meta_dict' in self.summary_spec:   
                # Custom Meta Dictionary passed
                self.meta_param_recorder.merge_custom(self.summary_spec['meta_dict'])
            if 'configuration' in self.summary_labels:  
                # Setup for TensorBoard population
                self.summary_spec['meta_param_recorder_class'] = self.meta_param_recorder
            if 'print_configuration' in self.summary_labels: 
                # Print to STDOUT (TADO: optimize output)
                self.meta_param_recorder.text_output(format_type=1)
 
        # Init Model, this must follow the Summary Configuration section above to cary meta_param_recorder
        self.model = self.initialize_model()

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

        #TODO have to call preprocessing reset in model
        # for preprocessing in self.preprocessing.values():
        #     preprocessing.reset()

    def act(self, states, deterministic=False):
        """
        Return action(s) for given state(s). States preprocessing and exploration are applied if  
        configured accordingly.

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
        Observe experience from the environment to learn from. Optionally preprocesses rewards
        Child classes should call super to get the processed reward
        EX: terminal, reward = super()...

        Args:
            terminal: boolean indicating if the episode terminated after the observation.
            reward: scalar reward that resulted from executing the action.
        """
        self.current_terminal = terminal
        self.current_reward = reward

        if self.batched_observe is not None and self.batched_observe > 0:
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
