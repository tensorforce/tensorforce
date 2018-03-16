from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest
import pytest

from tensorforce import TensorForceError
from tensorforce.core.networks import LayeredNetwork
from tensorforce.models import DistributionModel
from tensorforce.tests.minimal_test import MinimalTest
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
import tensorflow as tf
import numpy as np
from tensorforce.util import SavableComponent
import os


class SavableNetwork(LayeredNetwork, SavableComponent):
    """
    Minimal implementation of a Network that can be saved and restored independently of the Model.
    """
    def get_savable_variables(self):
        return super(SavableNetwork, self).get_variables(include_nontrainable=False)

    def _get_base_variable_scope(self):
        return self.apply.variable_scope_name


def create_environment(spec):
    return MinimalTest(spec)


def create_agent(environment, network_spec):
    return PPOAgent(
        update_mode=dict(
            unit='episodes',
            batch_size=4,
            frequency=4
        ),
        memory=dict(
            type='latest',
            include_next_states=False,
            capacity=100
        ),
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        subsampling_fraction=0.3,
        optimization_steps=20,
        states=environment.states,
        actions=environment.actions,
        network=network_spec
    )


class TestModelSaveRestore(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()
        self._tmp_dir_path = str(tmpdir)
        print("Using %s" % (self._tmp_dir_path, ))

    def test_save_restore(self):
        environment_spec = {"float": ()}
        environment = create_environment(environment_spec)
        network_spec = [
            dict(type='dense', size=32)
        ]
        agent = create_agent(environment, network_spec)
        runner = Runner(agent=agent, environment=environment)

        runner.run(episodes=100)
        model_values = agent.model.session.run(agent.model.get_variables(
            include_submodules=True,
            include_nontrainable=False
        ))
        save_path = agent.model.save(directory=self._tmp_dir_path + "/model")
        print("Saved at: %s" % (save_path,))
        runner.close()

        agent = create_agent(environment, network_spec)
        agent.model.restore(directory="", file=save_path)
        restored_model_values = agent.model.session.run(agent.model.get_variables(
            include_submodules=True,
            include_nontrainable=False
        ))
        assert len(model_values) == len(restored_model_values)
        assert all([np.array_equal(v1, v2) for v1, v2 in zip(model_values, restored_model_values)])

        agent.close()

    def test_save_network(self):
        """
        Test to validate that calls to save and restore of a SavableComponent successfully save and restore the
        component's state.
        """

        environment_spec = {"float": ()}
        environment = create_environment(environment_spec)
        network_spec = dict(
            type=SavableNetwork,
            layers=[dict(type='dense', size=1)]
        )
        agent = create_agent(environment, network_spec)
        assert isinstance(agent.model.network, SavableComponent)

        runner = Runner(agent=agent, environment=environment)
        runner.run(episodes=100)

        network_values = agent.model.session.run(agent.model.network.get_variables())
        distribution = next(iter(agent.model.distributions.values()))
        distribution_values = agent.model.session.run(distribution.get_variables())
        save_path = self._tmp_dir_path + "/network"
        agent.model.save_component(component_name=DistributionModel.COMPONENT_NETWORK, save_path=save_path)
        runner.close()

        assert os.path.isfile(save_path + ".data-00000-of-00001")
        assert os.path.isfile(save_path + ".index")

        agent = create_agent(environment, network_spec)
        agent.model.restore_component(component_name=DistributionModel.COMPONENT_NETWORK, save_path=save_path)

        # Ensure only the network variables are loaded
        restored_network_values = agent.model.session.run(agent.model.network.get_variables(include_nontrainable=True))
        distribution = next(iter(agent.model.distributions.values()))
        restored_distribution_values = agent.model.session.run(distribution.get_variables())

        assert len(restored_network_values) == len(network_values)
        assert all([np.array_equal(v1, v2) for v1, v2 in zip(network_values, restored_network_values)])
        assert len(restored_distribution_values) == len(distribution_values)
        assert not all([np.array_equal(v1, v2) for v1, v2 in zip(distribution_values, restored_distribution_values)])

        agent.close()
        environment.close()

    def test_pretrain_network(self):
        """
        Simulates training outside of Tensorforce and then loading the parameters in the agent's network.
        """

        environment_spec = {"float": ()}
        environment = create_environment(environment_spec)
        size = environment.states["shape"]
        output_size = 1
        save_path = self._tmp_dir_path + "/network"

        g = tf.Graph()
        with g.as_default():
            x = tf.placeholder(dtype=environment.states["type"], shape=[None, size])
            layer = tf.layers.Dense(units=output_size)
            y = layer(x)
            y_ = tf.placeholder(dtype=environment.states["type"], shape=[None, output_size])
            loss = tf.losses.mean_squared_error(y_, y)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            train_step = optimizer.minimize(loss)
            batch_size = 64
            with tf.Session(graph=g) as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(100):
                    batch = np.random.random([batch_size, size])
                    correct = np.ones(shape=[batch.shape[0], output_size])
                    loss_value, _ = sess.run([loss, train_step], {x: batch, y_: correct})
                    if epoch % 10 == 0:
                        print("epoch %d: %f" % (epoch, loss_value))
                var_map = {
                    "dense0/apply/linear/apply/W:0": layer.kernel,
                    "dense0/apply/linear/apply/b:0": layer.bias
                }
                saver = tf.train.Saver(var_list=var_map)
                saver.save(sess=sess, write_meta_graph=False, save_path=save_path)

        network_spec = dict(
            type=SavableNetwork,
            layers=[dict(type='dense', size=output_size)],
        )
        agent = create_agent(environment, network_spec)
        agent.model.restore_component(component_name=agent.model.COMPONENT_NETWORK, save_path=save_path)
        agent.close()

    def test_non_savable_component(self):
        environment_spec = {"float": ()}
        environment = create_environment(environment_spec)
        network_spec = [dict(type='dense', size=32)]
        agent = create_agent(environment, network_spec)
        expected_message = "Component network must implement SavableComponent but is "

        with pytest.raises(TensorForceError) as excinfo:
            agent.model.restore_component(component_name="network", save_path=self._tmp_dir_path + "/network")
        assert expected_message in str(excinfo.value)

        with pytest.raises(TensorForceError) as excinfo:
            agent.model.save_component(component_name="network", save_path=self._tmp_dir_path + "/network")
        assert expected_message in str(excinfo.value)

        with pytest.raises(TensorForceError) as excinfo:
            agent.model.restore_component(component_name="non-existent", save_path=self._tmp_dir_path + "/network")
        assert "Component non-existent must implement SavableComponent but is None" == str(excinfo.value)

        agent.close()
