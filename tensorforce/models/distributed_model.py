"""
Model for distributed tensorflow. Not generic yet, testing with vpg.

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from tensorforce.config import create_config
from tensorforce.models import LinearValueFunction
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.models.policies import CategoricalOneHotPolicy
from tensorforce.models.policies import GaussianPolicy
from tensorforce.util.config_util import get_function
from tensorforce.util.experiment_util import global_seed
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
        self.saver = None
        self.config = create_config(config, default=self.default_config)
        self.scope = scope
        self.batch_size = self.config.batch_size
        self.action_count = self.config.actions
        self.use_gae = self.config.use_gae
        self.gae_lambda = self.config.gae_lambda

        self.gamma = self.config.gamma
        self.continuous = self.config.continuous
        self.normalize_advantage = self.config.normalise_advantage

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        # This is the scope used to prefix variable creation for distributed TensorFlow
        self.batch_shape = [None]

        self.deterministic_mode = config.get('deterministic_mode', False)

        self.alpha = config.get('alpha', 0.001)

        self.worker_device = "/job:worker/task:{}/cpu:0".format(task_index)

        with tf.device(tf.train.replica_device_setter(1, worker_device=self.worker_device)):
            with tf.variable_scope("global"):
                # Dummy input, not used
                self.global_state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")
                self.global_network = NeuralNetwork(self.config.network_layers, self.global_state,
                                                   scope=scope + 'global_model')
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

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

    def create_training_operations(self):
        """
        Currently a duplicate of the pg model logic, to be made generic later to allow
        all models to be executed asynchronously/distributed seamlessly.

        """
        # TODO rewrite model logic so core update logic can be composed into
        # TODO distributed logic

        with tf.device(self.worker_device):
            with tf.variable_scope("local"):
                self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")

                self.local_network = NeuralNetwork(self.config.network_layers, self.state,
                                                    scope=self.scope + 'local_model')

            self.actions = tf.placeholder(tf.float32, [None, self.action_count], name='actions')
            self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
            self.prev_action_means = tf.placeholder(tf.float32, [None, self.action_count], name='prev_actions')

            if self.continuous:
                self.policy = GaussianPolicy(self.local_network, self.session, self.state, self.random,
                                             self.action_count, 'gaussian_policy')
                self.prev_action_log_stds = tf.placeholder(tf.float32, [None, self.action_count])

                self.prev_dist = dict(policy_output=self.prev_action_means,
                                      policy_log_std=self.prev_action_log_stds)

            else:
                self.policy = CategoricalOneHotPolicy(self.local_network, self.session, self.state, self.random,
                                                      self.action_count, 'categorical_policy')
                self.prev_dist = dict(policy_output=self.prev_action_means)

            # Probability distribution used in the current policy
            self.dist = self.policy.get_distribution()

            self.baseline_value_function = LinearValueFunction()
            self.log_probabilities = self.dist.log_prob(self.policy.get_policy_variables(), self.actions)

            # Concise: Get log likelihood of actions, weigh by advantages, compute gradient on that
            self.loss = -tf.reduce_mean(self.log_probabilities * self.advantage, name="loss_op")

            self.optimize_op = self.optimizer.minimize(self.loss)

    def get_action(self, state):
        raise NotImplementedError

    def update(self, batch):
        pass

    def get_global_step(self):
        """
        Returns global step to coordinator.
        :return:
        """
        return self.session.run(self.global_step)

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
