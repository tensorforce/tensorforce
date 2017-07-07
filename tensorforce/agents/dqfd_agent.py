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

"""
Deep Q-learning from demonstration. This agent pre-trains from demonstration data.
 
Original paper: 'Learning from Demonstrations for Real World Reinforcement Learning'

https://arxiv.org/abs/1704.03732
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

from tensorforce import util
from tensorforce.agents import Agent
from tensorforce.core.memories import memories
from tensorforce.models import DQFDModel


class DQFDAgent(Agent):
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
    * `optimizer_args`: list of arguments for optimizer.
    * `optimizer_kwargs`: dict of keyword arguments for optimizer.
    * `device`: string of tensorflow device name.
    * `tf_saver`: boolean whether to save model parameters.
    * `tf_summary`: boolean indicating whether to use tensorflow summary file writer.
    * `log_level`: string containing logleve (e.g. 'info').
    * `distributed`: boolean indicating whether to use distributed tensorflow.
    * `global_model`: global model.
    * `session`: session to use.


    The `DQFDAgent` class additionally requires the following parameters:


    * `batch_size`: integer of the batch size.
    * `memory_capacity`: integer of maximum experiences to store.
    * `memory`: string indicating memory type ('replay' or 'prioritized_replay').
    * `memory_args`: list of arguments to pass to replay memory constructor.
    * `memory_kwargs`: list of keyword arguments to pass to replay memory constructor.
    * `min_replay_size`: integer of minimum replay size before the first update.
    * `update_rate`: float of the update rate (e.g. 0.25 = every 4 steps).
    * `target_network_update_rate`: float of target network update rate (e.g. 0.01 = every 100 steps).
    * `use_target_network`: boolean indicating whether to use a target network.
    * `update_repeat`: integer of how many times to repeat an update.
    * `update_target_weight`: float of update target weight (tau parameter).
    * `expert_sampling_ratio`: float, ratio of expert data used at runtime to train from.
    * `supervised_weight`: float, weight of large margin classifier loss.
    * `expert_margin`: float of difference in Q-values between expert action and other actions enforced by the large margin function.
    * `clip_gradients`: float of maximum values for gradients before clipping.


    """

    name = 'DQFDAgent'
    model = DQFDModel
    default_config = dict(
        expert_sampling_ratio=0.01,
        batch_size=32,
        memory_capacity=1000000,
        memory='replay',
        memory_args=None,
        memory_kwargs=None,
        update_rate=0.25,
        target_network_update_rate=0.0001,
        min_replay_size=5e4,
        deterministic_mode=False,
        use_target_network=False,
        update_repeat=1
    )

    def __init__(self, config):
        """
        
        :param config: 
        :param scope: 
        """
        config.default(DQFDAgent.default_config)
        super(DQFDAgent, self).__init__(config)

        memory = util.function(config.memory, memories)
        args = config.optimizer_args or ()
        kwargs = config.optimizer_kwargs or {}

        # This is the online memory
        self.replay_memory = memory(config.memory_capacity, config.states, config.actions, *args, **kwargs)

        # This is the demonstration memory that we will fill with observations before starting
        # the main training loop
        # TODO we might want different sizes for these memories -> add config param
        self.demo_memory = memory(config.memory_capacity, config.states, config.actions, *args, **kwargs)
        self.step_count = 0

        # Called p in paper, controls ratio of expert vs online training samples
        self.expert_sampling_ratio = config.expert_sampling_ratio

        self.update_repeat = config.update_repeat
        self.batch_size = config.batch_size

        # p = n_demo / (n_demo + n_replay) => n_demo  = p * n_replay / (1 - p)
        self.demo_batch_size = int(self.expert_sampling_ratio * self.batch_size / \
                               (1.0 - self.expert_sampling_ratio))
        self.update_steps = int(round(1 / config.update_rate))
        self.use_target_network = config.use_target_network

        if self.use_target_network:
            self.target_update_steps = int(round(1 / config.target_network_update_rate))

        self.min_replay_size = config.min_replay_size

    def add_demo_observation(self, state, action, reward, terminal):
        """Adds observations to demo memory. 

        Args:
            state: 
            action: 
            reward: 
            terminal: 

        Returns:

        """
        if self.unique_state:
            state = dict(state=state)
        if self.unique_action:
            action = dict(action=action)

        self.demo_memory.add_experience(state, action, reward, terminal, internal=self.internal)

    def pre_train(self, steps=1):
        """Computes pretrain updates.
        
        Args:
            steps: Number of updates to execute.

        Returns:

        """
        for _ in xrange(steps):
            # Sample from demo memory
            batch = self.demo_memory.get_batch(self.batch_size)

            # Update using both double Q-learning and supervised double_q_loss
            self.model.pre_train_update(batch)

    def observe(self, state, action, reward, terminal):
        """Adds observations, updates via sampling from memories according to update rate.
        DQFD samples from the online replay memory and the demo memory with
        the fractions controlled by a hyper parameter p called 'expert sampling ratio.
        
        Args:
            state: 
            action: 
            reward: 
            terminal: 

        Returns:

        """
        if self.unique_state:
            state = dict(state=state)
        if self.unique_action:
            action = dict(action=action)

        self.replay_memory.add_experience(state, action, reward, terminal, internal=self.internal)

        self.step_count += 1

        if self.step_count >= self.min_replay_size and self.step_count % self.update_steps == 0:
            for _ in xrange(self.update_repeat):
                # Sample batches according to expert sampling ratio
                # In the paper, p is given as p = n_demo / (n_replay + n_demo)
                demo_batch = self.demo_memory.get_batch(self.demo_batch_size)
                online_batch = self.replay_memory.get_batch(self.batch_size)

                self.model.pre_train_update(batch=demo_batch)
                self.model.update(batch=online_batch)

        if self.step_count >= self.min_replay_size and self.use_target_network \
                and self.step_count % self.target_update_steps == 0:
            self.model.update_target_network()

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
