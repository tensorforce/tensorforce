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
            states_spec,
            actions_spec,
            batched_observe=1000,
            scope='dqn',
            # parameters specific to LearningAgents
            summary_spec=None,
            network_spec=None,
            discount=0.99,
            device=None,
            session_config=None,
            saver_spec=None,
            distributed_spec=None,
            optimizer=None,
            variable_noise=None,
            states_preprocessing_spec=None,
            explorations_spec=None,
            reward_preprocessing_spec=None,
            distributions_spec=None,
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

        # TensorFlow summaries & Configuration Meta Parameter Recorder options
        self.summary_spec = summary_spec
        if self.summary_spec is None:
            self.summary_labels = set()
        else:
            self.summary_labels = set(self.summary_spec.get('labels', ()))

        self.meta_param_recorder = None

        # if 'configuration' in self.summary_labels or 'print_configuration' in self.summary_labels:
        if any(k in self.summary_labels for k in ['configuration', 'print_configuration']):
            self.meta_param_recorder = MetaParameterRecorder(inspect.currentframe())
            if 'meta_dict' in self.summary_spec:
                # Custom Meta Dictionary passed
                self.meta_param_recorder.merge_custom(self.summary_spec['meta_dict'])
            if 'configuration' in self.summary_labels:
                # Setup for TensorBoard population
                self.summary_spec['meta_param_recorder_class'] = self.meta_param_recorder
            if 'print_configuration' in self.summary_labels:
                # Print to STDOUT (TODO: optimize output)
                self.meta_param_recorder.text_output(format_type=1)

        if network_spec is None:
            raise TensorForceError("No network_spec provided.")
        self.network_spec = network_spec

        self.discount = discount
        self.device = device
        self.session_config = session_config
        self.saver_spec = saver_spec
        self.distributed_spec = distributed_spec

        if optimizer is None:
            self.optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )
        else:
            self.optimizer = optimizer

        self.variable_noise = variable_noise
        self.states_preprocessing_spec = states_preprocessing_spec
        self.explorations_spec = explorations_spec
        self.reward_preprocessing_spec = reward_preprocessing_spec
        self.distributions_spec = distributions_spec
        self.entropy_regularization = entropy_regularization

        super(LearningAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe,
            scope=scope
        )

