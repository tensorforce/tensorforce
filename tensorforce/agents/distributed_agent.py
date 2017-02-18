"""
Generic agent for distributed training
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.config import create_config
from tensorforce.models.distributed_model import DistributedModel


class DistributedAgent(object):
    name = 'DistributedAgent'
    default_config = {}

    model_ref = None

    def __init__(self, config, scope, task_index):
        self.config = create_config(config, default=self.default_config)
        self.model = DistributedModel(config, scope, task_index)

    def increment_global_step(self):
        self.model.get_global_step()

    def get_action(self, state):
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
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def __str__(self):
        return self.name
