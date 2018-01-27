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

from tensorforce.agents import LearningAgent
from tensorforce.models import PGProbRatioModel


class TRPOAgent(LearningAgent):
    """
    Trust Region Policy Optimization ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) agent.
    """

    def __init__(
        self,
        states,
        actions,
        network,
        batched_observe=True,
        batching_capacity=1000,
        scope='trpo',
        device=None,
        saver=None,
        summarizer=None,
        distributed=None,
        variable_noise=None,
        states_preprocessing=None,
        actions_exploration=None,
        reward_preprocessing=None,
        update_mode=None,
        memory=None,
        discount=0.99,
        distributions=None,
        entropy_regularization=None,
        # parameters specific to BatchAgents
        batch_size=1000,
        keep_last_timestep=True,
        # parameters specific to trust-region pol.opt. Agents
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None,
        likelihood_ratio_clipping=None,
        learning_rate=1e-3,
        cg_max_iterations=20,
        cg_damping=1e-3,
        cg_unroll_loop=False,
        ls_max_iterations=10,
        ls_accept_ratio=0.9,
        ls_unroll_loop=False
    ):
        """
        Creates a Trust Region Policy Optimization ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) agent.

        Args:
            states: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            network: List of layers specifying a neural network via layer types, sizes and optional arguments
                such as activation or regularisation. Full examples are in the examples/configs folder.
            device: Device string specifying model device.
            session_config: optional tf.ConfigProto with additional desired session configurations
            scope: TensorFlow scope, defaults to agent name (e.g. `dqn`).
            saver: Dict specifying automated saving. Use `directory` to specify where checkpoints are saved. Use
                either `seconds` or `steps` to specify how often the model should be saved. The `load` flag specifies
                if a model is initially loaded (set to True) from a file `file`.
            summary: Dict specifying summarizer for TensorBoard. Requires a 'directory' to store summarizer, `steps`
                or `seconds` to specify how often to save summarizer, and a list of `labels` to indicate which values
                to export, e.g. `losses`, `variables`. Consult neural network class and model for all available labels.
            distributed: Dict specifying distributed functionality. Use `parameter_server` and `replica_model`
                Boolean flags to indicate workers and parameter servers. Use a `cluster` key to pass a TensorFlow
                cluster spec.
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
            likelihood_ratio_clipping: Optional clipping of likelihood ratio between old and new policy.
            learning_rate: Learning rate which may be interpreted differently according to optimizer, e.g. a natural
                gradient optimizer interprets the learning rate as the max kl-divergence between old and updated policy.
            cg_max_iterations: Int > 0 specifying conjugate gradient iterations, typically 10-20 are sufficient to
                find effective approximate solutions.
            cg_damping: Conjugate gradient damping value to increase numerical stability.
            cg_unroll_loop: Boolean indicating whether loop unrolling in TensorFlow is to be used which seems to
                impact performance negatively at this point, default False.
        """

        # Update mode
        if update_mode is None:
            update_mode = dict(
                unit='episodes',
                batch_size=10,
                frequency=10
            )
        elif 'unit' in update_mode:
            assert update_mode['unit'] == 'episodes'
        else:
            update_mode['unit'] = 'episodes'

        # Memory
        if memory is None:
            # Assumed episode length of 1000 timesteps.
            memory = dict(
                type='latest',
                include_next_states=False,
                capacity=(1000 * update_mode['batch_size'])
            )
        else:
            assert not memory['include_next_states']

        # Optimizer
        optimizer = dict(
            type='optimized_step',
            optimizer=dict(
                type='natural_gradient',
                learning_rate=learning_rate,
                cg_max_iterations=cg_max_iterations,
                cg_damping=cg_damping,
                cg_unroll_loop=cg_unroll_loop,
            ),
            ls_max_iterations=ls_max_iterations,
            ls_accept_ratio=ls_accept_ratio,
            ls_mode='exponential',  # !!!!!!!!!!!!!
            ls_parameter=0.5,  # !!!!!!!!!!!!!
            ls_unroll_loop=ls_unroll_loop
        )

        self.baseline_mode = baseline_mode
        self.baseline = baseline
        self.baseline_optimizer = baseline_optimizer
        self.gae_lambda = gae_lambda
        self.likelihood_ratio_clipping = likelihood_ratio_clipping

        super(TRPOAgent, self).__init__(
            states=states,
            actions=actions,
            network=network,
            batched_observe=batched_observe,
            batching_capacity=batching_capacity,
            scope=scope,
            device=device,
            saver=saver,
            summarizer=summarizer,
            distributed=distributed,
            variable_noise=variable_noise,
            states_preprocessing=states_preprocessing,
            actions_exploration=actions_exploration,
            reward_preprocessing=reward_preprocessing,
            update_mode=update_mode,
            memory=memory,
            optimizer=optimizer,
            discount=discount,
            distributions=distributions,
            entropy_regularization=entropy_regularization
        )

    def initialize_model(self):
        return PGProbRatioModel(
            states=self.states,
            actions=self.actions,
            scope=self.scope,
            device=self.device,
            saver=self.saver,
            summarizer=self.summarizer,
            distributed=self.distributed,
            batching_capacity=self.batching_capacity,
            discount=self.discount,
            variable_noise=self.variable_noise,
            states_preprocessing=self.states_preprocessing,
            actions_exploration=self.actions_exploration,
            reward_preprocessing=self.reward_preprocessing,
            update_mode=self.update_mode,
            memory=self.memory,
            optimizer=self.optimizer,
            network=self.network,
            distributions=self.distributions,
            entropy_regularization=self.entropy_regularization,
            baseline_mode=self.baseline_mode,
            baseline=self.baseline,
            baseline_optimizer=self.baseline_optimizer,
            gae_lambda=self.gae_lambda,
            likelihood_ratio_clipping=self.likelihood_ratio_clipping
        )
