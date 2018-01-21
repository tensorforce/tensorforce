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

import inspect

from tensorforce import TensorForceError
from tensorforce.agents.agent import Agent
from tensorforce.meta_parameter_recorder import MetaParameterRecorder


class LearningAgent(Agent):
    """
    An Agent that actually learns by optimizing the parameters of its tensorflow model.

    """

    def __init__(
        self,
        states,
        actions,
        network,
        batched_observe=True,
        batching_capacity=1000,
        scope='learning-agent',
        device=None,
        saver=None,
        summaries=None,
        distributed=None,
        variable_noise=None,
        states_preprocessing=None,
        actions_exploration=None,
        reward_preprocessing=None,
        update_mode=None,
        memory=None,
        optimizer=None,
        discount=0.99,
        distributions=None,
        entropy_regularization=None
    ):
        """
        Initializes the learning agent.

        Args:
            summary_spec: Dict specifying summaries for TensorBoard. Requires a 'directory' to store summaries, `steps`
                or `seconds` to specify how often to save summaries, and a list of `labels` to indicate which values
                to export, e.g. `losses`, `variables`. Consult neural network class and model for all available labels.
            network_spec: List of layers specifying a neural network via layer types, sizes and optional arguments
                such as activation or regularisation. Full examples are in the examples/configs folder.
            discount (float): The reward discount factor.
            device: Device string specifying model device.
            session_config: optional tf.ConfigProto with additional desired session configurations
            saver_spec: Dict specifying automated saving. Use `directory` to specify where checkpoints are saved. Use
                either `seconds` or `steps` to specify how often the model should be saved. The `load` flag specifies
                if a model is initially loaded (set to True) from a file `file`.
            distributed_spec: Dict specifying distributed functionality. Use `parameter_server` and `replica_model`
                Boolean flags to indicate workers and parameter servers. Use a `cluster_spec` key to pass a TensorFlow
                cluster spec.
            optimizer: Dict specifying optimizer type and its optional parameters, typically a `learning_rate`.
                Available optimizer types include standard TensorFlow optimizers, `natural_gradient`,
                and `evolutionary`. Consult the optimizer test or example configurations for more.
            variable_noise: Experimental optional parameter specifying variable noise (NoisyNet).
            states_preprocessing_spec: Optional list of states preprocessors to apply to state
                (e.g. `image_resize`, `greyscale`).
            explorations_spec: Optional dict specifying action exploration type (epsilon greedy
                or Gaussian noise).
            reward_preprocessing_spec: Optional dict specifying reward preprocessing.
            distributions_spec: Optional dict specifying action distributions to override default distribution choices.
                Must match action names.
            entropy_regularization: Optional positive float specifying an entropy regularization value.
        """
        if network is None:
            raise TensorForceError("No network provided.")

        self.scope = scope
        self.device = device
        self.saver = saver
        self.summaries = summaries
        self.distributed = distributed
        self.variable_noise = variable_noise
        self.states_preprocessing = states_preprocessing
        self.actions_exploration = actions_exploration
        self.reward_preprocessing = reward_preprocessing
        self.update_mode = update_mode
        self.memory = memory
        self.optimizer = optimizer
        self.discount = discount
        self.network = network
        self.distributions = distributions
        self.entropy_regularization = entropy_regularization

        # TensorFlow summaries & Configuration Meta Parameter Recorder options
        if self.summaries is None:
            summary_labels = set()
        else:
            summary_labels = set(self.summaries.get('labels', ()))

        self.meta_param_recorder = None

        # if 'configuration' in self.summary_labels or 'print_configuration' in self.summary_labels:
        if any(k in summary_labels for k in ['configuration', 'print_configuration']):
            self.meta_param_recorder = MetaParameterRecorder(inspect.currentframe())
            if 'meta_dict' in self.summaries:
                # Custom Meta Dictionary passed
                self.meta_param_recorder.merge_custom(self.summaries['meta_dict'])
            if 'configuration' in summary_labels:
                # Setup for TensorBoard population
                self.summaries['meta_param_recorder_class'] = self.meta_param_recorder
            if 'print_configuration' in summary_labels:
                # Print to STDOUT (TODO: optimize output)
                self.meta_param_recorder.text_output(format_type=1)

        super(LearningAgent, self).__init__(
            states=states,
            actions=actions,
            batched_observe=batched_observe,
            batching_capacity=batching_capacity
        )

    def import_experience(self, experiences):
        """
        Imports experiences.

        Args:
            experiences: 
        """
        if isinstance(experiences, dict):
            if self.unique_state:
                experiences['states'] = dict(state=experiences['states'])
            if self.unique_action:
                experiences['actions'] = dict(action=experiences['actions'])

            self.model.import_experience(**experiences)

        else:
            if self.unique_state:
                states = dict(state=list())
            else:
                states = {name: list() for name in experiences[0]['states']}
            internals = [list() for _ in experiences[0]['internals']]
            if self.unique_action:
                actions = dict(action=list())
            else:
                actions = {name: list() for name in experiences[0]['actions']}
            terminal = list()
            reward = list()

            for experience in experiences:
                if self.unique_state:
                    states['state'].append(experience['states'])
                else:
                    for name, state in states.items():
                        state.append(experience['states'][name])
                for n, internal in enumerate(internals):
                    internal.append(experience['internals'][n])
                if self.unique_action:
                    actions['action'].append(experience['actions'])
                else:
                    for name, action in actions.items():
                        action.append(experience['actions'][name])
                terminal.append(experience['terminal'])
                reward.append(experience['reward'])

            self.model.import_experience(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward
            )
