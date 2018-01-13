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
import numpy as np

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.core.memories import Replay
from tensorforce.models import QDemoModel


class DQFDAgent(Agent):
    """
    Deep Q-learning from demonstration (DQFD) agent ([Hester et al., 2017](https://arxiv.org/abs/1704.03732)).
    This agent uses DQN to pre-train from demonstration data via an additional supervised loss term.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        network,
        device=None,
        session_config=None,
        scope='dqfd',
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
        target_sync_frequency=10000,
        target_update_weight=1.0,
        huber_loss=None,
        batched_observe=None,
        batch_size=32,
        update_frequency=4,
        # first_update=10000,
        # repeat_update=1
        expert_margin=0.5,
        supervised_weight=0.1,
        demo_memory_capacity=10000,
        demo_sampling_ratio=0.2
    ):
        """
        Deep Q-learning from demonstration (DQFD) agent ([Hester et al., 2017](https://arxiv.org/abs/1704.03732)).
        This agent uses DQN to pre-train from demonstration data in combination with a supervised loss.

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
            states_preprocessing_spec: Optional list of states preprocessors to apply to state  
                (e.g. `image_resize`, `grayscale`).
            actions_exploration_spec: Optional dict specifying action exploration type (epsilon greedy  
                or Gaussian noise).
            reward_preprocessing_spec: Optional dict specifying reward preprocessing.
            distributions_spec: Optional dict specifying action distributions to override default distribution choices.
                Must match action names.
            entropy_regularization: Optional positive float specifying an entropy regularization value.
            target_sync_frequency: Interval between optimization calls synchronizing the target network.
            target_update_weight: Update weight, 1.0 meaning a full assignment to target network from training network.
            huber_loss: Optional flat specifying Huber-loss clipping.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
            batch_size: Int specifying batch size used to sample from memory. Should be smaller than memory size.
            memory: Dict describing memory via `type` (e.g. `replay`) and `capacity`.
            first_update: Int describing at which time step the first update is performed. Should be larger
                than batch size.
            update_frequency: Int specifying number of observe steps to perform until an update is executed.
            repeat_update: Int specifying how many update steps are performed per update, where each update step implies
                sampling a batch from the memory and passing it to the model.
            expert_margin: Positive float specifying enforced supervised margin between expert action Q-value and other
                Q-values.
            supervised_weight: Weight of supervised loss term.
            demo_memory_capacity: Int describing capacity of expert demonstration memory.
            demo_sampling_ratio: Runtime sampling ratio of expert data.
        """
        if network is None:
            raise TensorForceError("No network provided.")

        if memory is None:
            memory = dict(
                type='replay',
                include_next_states=True,
                capacity=(1000 * batch_size)  # capacity of 1000 batches
            )
        else:
            assert memory['include_next_states']
        update_spec = dict(
            mode='timesteps',
            batch_size=batch_size,
            frequency=update_frequency
        )
        if optimizer is None:
            optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )

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
        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight
        self.double_q_model = True
        self.huber_loss = huber_loss
        self.expert_margin = expert_margin
        self.supervised_weight = supervised_weight

        self.demo_memory_capacity = demo_memory_capacity
        # The demo_sampling_ratio, called p in paper, controls ratio of expert vs online training samples
        # p = n_demo / (n_demo + n_replay) => n_demo  = p * n_replay / (1 - p)
        self.demo_batch_size = int(demo_sampling_ratio * batch_size / (1.0 - demo_sampling_ratio))
        assert self.demo_batch_size > 0, 'Check DQFD sampling parameters to ensure ' \
                                         'demo_batch_size is positive. (Calculated {} based on current' \
                                         ' parameters)'.format(self.demo_batch_size)

        # This is the demonstration memory that we will fill with observations before starting
        # the main training loop
        super(DQFDAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe
        )

    def initialize_model(self):
        return QDemoModel(
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
            target_sync_frequency=self.target_sync_frequency,
            target_update_weight=self.target_update_weight,
            double_q_model=self.double_q_model,
            huber_loss=self.huber_loss,
            expert_margin=self.expert_margin,
            supervised_weight=self.supervised_weight,
            demo_memory_capacity=self.demo_memory_capacity,
            demo_batch_size=self.demo_batch_size
        )

    def observe(self, reward, terminal):
        """
        Adds observations, updates via sampling from memories according to update rate.
        DQFD samples from the online replay memory and the demo memory with
        the fractions controlled by a hyper parameter p called 'expert sampling ratio.

        Args:
            reward:
            terminal:
        """
        super(DQFDAgent, self).observe(reward=reward, terminal=terminal)
        # TODO Where are these parameters?
        if self.timestep >= self.first_update and self.timestep % self.update_frequency == 0:
            for _ in xrange(self.repeat_update):
                self.model.demonstration_update()

    def import_demonstrations(self, demonstrations):
        """
        Imports demonstrations, i.e. expert observations. Note that for large numbers of observations,
        set_demonstrations is more appropriate, which directly sets memory contents to an array an expects
        a different layout.

        Args:
            demonstrations: List of observation dicts
        """
        for observation in demonstrations:
            if self.unique_state:
                state = dict(state=observation['states'])
            else:
                state = observation['states']
            if self.unique_action:
                action = dict(action=observation['actions'])
            else:
                action = observation['actions']
            # TODO this is undesirable now because it implies one session call per addition of sample
            # self.demo_memory.add_observation(
            #     states=state,
            #     internals=observation['internals'],
            #     actions=action,
            #     terminal=observation['terminal'],
            #     reward=observation['reward']
            # )

    def set_demonstrations(self, batch):
        """
        Set all demonstrations from batch data. Expects a dict wherein each value contains an array
        containing all states, actions, rewards, terminals and internals respectively.

        Args:
            batch:

        """
        self.model.set_demo_memory(
            states=batch['states'],
            internals=batch['internals'],
            actions=batch['actions'],
            terminal=batch['terminal'],
            reward=batch['reward']
        )
        # self.demo_memory.set_memory(
        #     states=batch['states'],
        #     internals=batch['internals'],
        #     actions=batch['actions'],
        #     terminal=batch['terminal'],
        #     reward=batch['reward']
        # )

    def pretrain(self, steps):
        """
        Computes pretrain updates.

        Args:
            steps: Number of updates to execute.

        """
        for _ in xrange(steps):
            self.model.demonstration_update()
