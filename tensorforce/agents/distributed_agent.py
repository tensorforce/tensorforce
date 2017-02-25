# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Generic agent for distributed training
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import defaultdict
import numpy as np
from copy import deepcopy

from tensorforce.config import create_config
from tensorforce.models.distributed_model import DistributedModel


class DistributedAgent(object):
    name = 'DistributedAgent'
    default_config = {}

    model_ref = None

    def __init__(self, config, scope, task_index, cluster_spec):
        self.config = create_config(config, default=self.default_config)
        self.current_episode = defaultdict(list)

        self.continuous = self.config.continuous
        self.batch = Batch(self.config)
        self.model = DistributedModel(config, scope, task_index, cluster_spec)

    def increment_global_step(self):
        self.model.get_global_step()

    def add_observation(self, state, action, reward, terminal):
        """
        Adds an observation and performs a pg update if the necessary conditions
        are satisfied, i.e. if one batch of experience has been collected as defined
        by the batch size.

        In particular, note that episode control happens outside of the agent since
        the agent should be agnostic to how the training data is created.

        :param state:
        :param action:
        :param reward:
        :param terminal:
        :return:
        """

        self.batch.add_observation(state, action, reward, terminal)

    def update(self, batch):
        """
        Updates the model using the given batch of experiences.

        """

        # Instead of calling update when a batch size is reached, explicitly called by
        # thread runner
        self.model.update(deepcopy(batch))
        self.batch = Batch(self.config)

    def get_action(self, *args, **kwargs):
        """
        Executes one reinforcement learning step.

        :param state: Observed state tensor
        :param episode: Optional, current episode
        :return: Which action to take
        """
        action, outputs = self.model.get_action(*args, **kwargs)
        #print(outputs)
        # Cache last action in case action is used multiple times in environment
        self.batch.last_action_means = outputs['policy_output']
        self.batch.last_action = action

        # print('action =' + str(action))

        if self.continuous:
            self.batch.last_action_log_std = outputs['policy_log_std']
        else:
            action = np.argmax(action)

        # print('action selected' + str(action))
        return action

    def load_model(self, path):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def set_session(self, session):
        self.model.set_session(session)

    def __str__(self):
        return self.name


class Batch(object):
    """
    Helper object for queue management.
    """

    def __init__(self, config):
        self.config = config
        self.current_episode = defaultdict(list)

        self.current_batch = []
        self.last_action = None
        self.last_action_means = None
        self.last_action_log_std = None
        self.continuous = self.config.continuous

    def add_observation(self, state, action, reward, terminal):
        """
        Adds an observation and performs a pg update if the necessary conditions
        are satisfied, i.e. if one batch of experience has been collected as defined
        by the batch size.

        In particular, note that episode control happens outside of the agent since
        the agent should be agnostic to how the training data is created.

        :param state:
        :param action:
        :param reward:
        :param terminal:
        :return:
        """

        self.current_episode['states'].append(state)
        self.current_episode['actions'].append(self.last_action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['action_means'].append(self.last_action_means)

        if self.continuous:
            self.current_episode['action_log_stds'].append(self.last_action_log_std)

        if terminal:
            # Batch could also end before episode is terminated
            self.current_episode['terminated'] = True

            # Transform into np arrays, append episode to batch, start new episode dict
            path = self.get_path()
            self.current_batch.append(path)
            self.current_episode = defaultdict(list)

    def extend(self, other):
        """
        Append another episode to this batch.
        :param other:
        :return:
        """
        # TODO this might not be correct, check episode/batch semantics
        self.current_batch.append(other)

    def get_path(self):
        """
        Finalises an episode and turns it into a dict pointing to numpy arrays.
        :return:
        """

        path = {'states': np.concatenate(np.expand_dims(self.current_episode['states'], 0)),
                'actions': np.array(self.current_episode['actions']),
                'terminated': self.current_episode['terminated'],
                'action_means': np.array(self.current_episode['action_means']),
                'rewards': np.array(self.current_episode['rewards'])}

        if self.continuous:
            path['action_log_stds'] = np.concatenate(self.current_episode['action_log_stds'])

        return path
