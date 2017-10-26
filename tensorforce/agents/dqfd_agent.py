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

from tensorforce.agents import MemoryAgent
from tensorforce.core.memories import Replay
from tensorforce.models import QDemoModel


class DQFDAgent(MemoryAgent):
    """
    Deep Q-learning from demonstration (DQFD) agent ([Hester et al., 2017](https://arxiv.org/abs/1704.03732)).
    This agent uses DQN to pre-train from demonstration data.

    Configuration:

    Each agent requires the following configuration parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    Each model requires the following configuration parameters:

    * `discount`: float of discount factor (gamma).
    * `learning_rate`: float of learning rate (alpha).
    * `optimizer`: string of optimizer to use (e.g. 'adam').
    * `device`: string of tensorflow device name.
    * `tf_summary`: string directory to write tensorflow summaries. Default None
    * `tf_summary_level`: int indicating which tensorflow summaries to create.
    * `tf_summary_interval`: int number of calls to get_action until writing tensorflow summaries on update.
    * `log_level`: string containing logleve (e.g. 'info').
    * `distributed`: boolean indicating whether to use distributed tensorflow.
    * `global_model`: global model.
    * `session`: session to use.


    The `DQFDAgent` class additionally requires the following parameters:


    * `batch_size`: integer of the batch size.
    * `memory_capacity`: integer of maximum experiences to store.
    * `memory`: string indicating memory type ('replay' or 'prioritized_replay').
    * `min_replay_size`: integer of minimum replay size before the first update.
    * `update_rate`: float of the update rate (e.g. 0.25 = every 4 steps).
    * `target_network_update_rate`: float of target network update rate (e.g. 0.01 = every 100 steps).
    * `use_target_network`: boolean indicating whether to use a target network.
    * `update_repeat`: integer of how many times to repeat an update.
    * `update_target_weight`: float of update target weight (tau parameter).
    * `demo_sampling_ratio`: float, ratio of expert data used at runtime to train from.
    * `supervised_weight`: float, weight of large margin classifier loss.
    * `expert_margin`: float of difference in Q-values between expert action and other actions enforced
                       by the large margin function.
    * `clip_loss`: float if not 0, uses the huber loss with clip_loss as the linear bound


    """
    default_config = dict(
        # Agent
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        # Model
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        discount=0.99,
        normalize_rewards=False,
        variable_noise=None,  # not documented!!!
        # DistributionModel
        distributions=None,  # not documented!!!
        entropy_regularization=None,
        # QModel
        target_sync_frequency=10000,  # not documented!!!
        target_update_weight=1.0,  # not documented!!!
        huber_loss=0.0,  # not documented!!!
        # Logging
        log_level='info',
        model_directory=None,
        save_frequency=600,  # TensorFlow default
        summary_labels=['total-loss'],
        summary_frequency=120,  # TensorFlow default
        # TensorFlow distributed configuration
        cluster_spec=None,
        parameter_server=False,
        task_index=0,
        device=None,
        local_model=False,
        replica_model=False,
        scope='dqfd'
    )

    def __init__(self, states_spec, actions_spec, network_spec, config):
        self.network_spec = network_spec
        config = config.copy()
        config.default(DQFDAgent.default_config)

        # DQFD always uses double dqn, which is a required key for a q-model.
        config.obligatory(double_dqn=True)
        self.target_update_frequency = config.target_update_frequency
        self.demo_memory_capacity = config.demo_memory_capacity

        # The demo_sampling_ratio, called p in paper, controls ratio of expert vs online training samples
        # p = n_demo / (n_demo + n_replay) => n_demo  = p * n_replay / (1 - p)
        self.demo_batch_size = int(config.demo_sampling_ratio * config.batch_size / (1.0 - config.demo_sampling_ratio))

        assert self.demo_batch_size > 0, 'Check DQFD sampling parameters to ensure ' \
                                         'demo_batch_size is positive. (Calculated {} based on current' \
                                         ' parameters)'.format(self.demo_batch_size)

        # This is the demonstration memory that we will fill with observations before starting
        # the main training loop
        self.demo_memory = Replay(self.demo_memory_capacity, self.states_spec, self.actions_spec)

        super(DQFDAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            config=config
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

        if self.timestep >= self.first_update and self.timestep % self.update_frequency == 0:
            for _ in xrange(self.repeat_update):
                batch = self.demo_memory.get_batch(self.demo_batch_size)
                self.model.demonstration_update(batch=batch)

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

            self.demo_memory.add_observation(
                states=state,
                internals=observation['internal'],
                actions=action,
                terminal=observation['terminal'],
                reward=observation['reward']
            )

    def set_demonstrations(self, batch):
        """
        Set all demonstrations from batch data. Expects a dict wherein each value contains an array
        containing all states, actions, rewards, terminals and internals respectively.

        Args:
            batch:

        """
        self.demo_memory.set_memory(
            states=batch['states'],
            internals=batch['internals'],
            actions=batch['actions'],
            terminal=batch['terminal'],
            reward=batch['reward']
        )

    def initialize_model(self, states_spec, actions_spec, config):
        return QDemoModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=self.network_spec,
            config=config
        )

    def pretrain(self, steps):
        """
        Computes pretrain updates.

        Args:
            steps: Number of updates to execute.

        """
        for _ in xrange(steps):
            # Sample from demo memory.
            batch = self.demo_memory.get_batch(batch_size=self.batch_size, next_states=True)

            # Update using both double Q-learning and supervised double_q_loss.
            self.model.demonstration_update(batch)
