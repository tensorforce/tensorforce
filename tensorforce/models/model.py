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
Models provide the general interface to TensorFlow functionality,
manages TensorFlow session and execution. In particular, a agent for reinforcement learning
always needs to provide a function that gives an action, and one to trigger updates.
A agent may use one more multiple neural networks and implement the update logic of a particular
RL algorithm.

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
import tensorflow as tf

from tensorforce.config import create_config
from tensorforce.util.config_util import get_function, log_levels
from tensorforce.util.exploration_util import exploration_mode


class Model(object):
    default_config = {}

    def __init__(self, config, scope):
        """
        
        :param config: Configuration parameters
        :param scope: TensorFlow scope
        """
        self.session = tf.Session()
        self.total_states = 0
        self.saver = None
        self.config = create_config(config, default=self.default_config)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_levels[config.loglevel])

        # This is the scope used to prefix variable creation for distributed TensorFlow
        self.scope = scope

        self.deterministic_mode = config.get('deterministic_mode', False)
        self.episode_length = tf.placeholder(tf.int32, (None,), name='episode_length')

        self.alpha = config.get('alpha', 0.001)

        optimizer = config.get('optimizer')
        if not optimizer:
            self.optimizer = tf.train.AdamOptimizer(self.alpha)
        else:
            args = config.get('optimizer_args', [])
            kwargs = config.get('optimizer_kwargs', {})
            optimizer_cls = get_function(optimizer)
            self.optimizer = optimizer_cls(self.alpha, *args, **kwargs)

        exploration = config.get('exploration')
        if not exploration:
            self.exploration = exploration_mode['constant'](self, 0)
        else:
            args = config.get('exploration_args', [])
            kwargs = config.get('exploration_kwargs', {})
            self.exploration = exploration_mode[exploration](self, *args, **kwargs)

    def get_action(self, state):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def get_variables(self):
        raise NotImplementedError

    def assign_variables(self, values):
        raise NotImplementedError

    def get_gradients(self):
        raise NotImplementedError

    def apply_gradients(self, grads_and_vars):
        raise NotImplementedError

    def load_model(self, path):
        self.saver.restore(self.session, path)

    def save_model(self, path):
        self.saver.save(self.session, path)
