"""
Generic model for distributed tensorflow.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce.config import create_config
from tensorforce.util.config_util import get_function
from tensorforce.util.exploration_util import exploration_mode

class DistributedModel(object):
    default_config = {}

    def __init__(self, config, scope, task_index):
        """

        A distributed model must synchronise local and global parameters. 

        :param config: Configuration parameters
        :param scope: TensorFlow scope
        """

        self.session = None
        self.total_states = 0
        self.saver = None
        self.config = create_config(config, default=self.default_config)

        # This is the scope used to prefix variable creation for distributed TensorFlow
        self.scope = scope
        self.batch_shape = [None]

        self.deterministic_mode = config.get('deterministic_mode', False)

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

    def set_session(self, session):
        self.session = session

        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError

    def update(self, batch):
        pass

    def increment_global_step(self):
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
