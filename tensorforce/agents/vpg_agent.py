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

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.models import PGLogProbModel


class VPGAgent(Agent):
    """
    Vanilla Policy Gradient agent as described by [Sutton et al. (1999)]
    (https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf).

    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        network,
        device=None,
        session_config=None,
        scope='vpg',
        saver_spec=None,
        summary_spec=None,
        distributed_spec=None,
        variable_noise=None,
        states_preprocessing=None,
        actions_exploration=None,
        reward_preprocessing=None,
        memory=None,
        update_spec=None,
        optimizer=None,
        discount=0.99,
        distributions=None,
        entropy_regularization=None,
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None,
        batched_observe=None,  # !!!!!!!!!!!!!
        batch_size=10,
        update_frequency=None
        # keep_last_timestep=True
    ):
        """
        Creates a vanilla policy gradient agent.

        Args:
            states_spec: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions_spec: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            network_spec: List of layers specifying a neural network via layer types, sizes and optional arguments
                such as activation or regularisation. Full examples are in the examples/configs folder.
            device: Device string specifying model device.
            session_config: optional tf.ConfigProto with additional desired session configurations
            scope: TensorFlow scope, defaults to agent name (e.g. `dqn`).
            saver_spec: Dict specifying automated saving. Use `directory` to specify where checkpoints are saved. Use
                either `seconds` or `steps` to specify how often the model should be saved. The `load` flag specifies
                if a model is initially loaded (set to True) from a file `file`.
            summary_spec: Dict specifying summaries for TensorBoard. Requires a 'directory' to store summaries, `steps`
                or `seconds` to specify how often to save summaries, and a list of `labels` to indicate which values
                to export, e.g. `losses`, `variables`. Consult neural network class and model for all available labels.
            distributed_spec: Dict specifying distributed functionality. Use `parameter_server` and `replica_model`
                Boolean flags to indicate workers and parameter servers. Use a `cluster_spec` key to pass a TensorFlow
                cluster spec.
            optimizer: Dict specifying optimizer type and its optional parameters, typically a `learning_rate`.
                Available optimizer types include standard TensorFlow optimizers, `natural_gradient`,
                and `evolutionary`. Consult the optimizer test or example configurations for more.
            discount: Float specifying reward discount factor.
            variable_noise: Experimental optional parameter specifying variable noise (NoisyNet).
            states_preprocessing: Optional list of states preprocessors to apply to state  
                (e.g. `image_resize`, `grayscale`).
            actions_exploration: Optional dict specifying action exploration type (epsilon greedy  
                or Gaussian noise).
            reward_preprocessing: Optional dict specifying reward preprocessing.
            distributions: Optional dict specifying action distributions to override default distribution choices.
                Must match action names.
            entropy_regularization: Optional positive float specifying an entropy regularization value.
            baseline_mode: String specifying baseline mode, `states` for a separate baseline per state, `network`
                for sharing parameters with the training network.
            baseline: Optional dict specifying baseline type (e.g. `mlp`, `cnn`), and its layer sizes. Consult
             examples/configs for full example configurations.
            baseline_optimizer: Optional dict specifying an optimizer and its parameters for the baseline
                following the same conventions as the main optimizer.
            gae_lambda: Optional float specifying lambda parameter for generalized advantage estimation.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
            batch_size: Int specifying number of samples collected via `observe` before an update is executed.
            keep_last_timestep: Boolean flag specifying whether last sample is kept, default True.
        """
        if network is None:
            raise TensorForceError("No network provided.")

        if memory is None:
            memory = dict(
                type='latest',
                include_next_states=False,
                capacity=(1000 * batch_size)  # assumed episode length of 1000
            )
        else:
            assert not memory['include_next_states']
        if update_frequency is None:
            update_frequency = batch_size
        update_spec = dict(
            mode='episodes',
            batch_size=batch_size,
            frequency=update_frequency
        )
        if optimizer is None:
            optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )

        # Model arguments
        self.device = device
        self.session_config = session_config
        self.scope = scope
        self.saver_spec = saver_spec
        self.summary_spec = summary_spec
        self.distributed_spec = distributed_spec
        self.variable_noise = variable_noise
        self.states_preprocessing = states_preprocessing
        self.actions_exploration = actions_exploration
        self.reward_preprocessing = reward_preprocessing
        self.memory = memory
        self.update_spec = update_spec
        self.optimizer = optimizer
        self.discount = discount
        self.network = network
        self.distributions = distributions
        self.entropy_regularization = entropy_regularization
        self.baseline_mode = baseline_mode
        self.baseline = baseline
        self.baseline_optimizer = baseline_optimizer
        self.gae_lambda = gae_lambda

        super(VPGAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe
        )

    def initialize_model(self):
        return PGLogProbModel(
            states_spec=self.states_spec,
            actions_spec=self.actions_spec,
            device=self.device,
            session_config=self.session_config,
            scope=self.scope,
            saver_spec=self.saver_spec,
            summary_spec=self.summary_spec,
            distributed_spec=self.distributed_spec,
            variable_noise=self.variable_noise,
            states_preprocessing=self.states_preprocessing,
            actions_exploration=self.actions_exploration,
            reward_preprocessing=self.reward_preprocessing,
            memory=self.memory,
            update_spec=self.update_spec,
            optimizer=self.optimizer,
            discount=self.discount,
            network=self.network,
            distributions=self.distributions,
            entropy_regularization=self.entropy_regularization,
            baseline_mode=self.baseline_mode,
            baseline=self.baseline,
            baseline_optimizer=self.baseline_optimizer,
            gae_lambda=self.gae_lambda
        )
