from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest
import pytest

from tensorforce import TensorForceError
from tensorforce.core.networks import LayeredNetwork
from tensorforce.environments import Environment
from tensorforce.models import DistributionModel
from .minimal_test import MinimalTest
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
import tensorflow as tf
import numpy as np
from tensorforce.util import SavableComponent
import os


class DummyEnv(Environment):
    """
    Dummy environment -> spits out random number for state and never terminates.
    """

    def __init__(self):
        super(DummyEnv, self).__init__()

    def __str__(self):
        return 'DummyEnv'

    def close(self):
        pass

    def reset(self):
        return np.random.uniform(size=[1])

    def execute(self, action):
        reward = np.random.uniform()
        terminal = False
        state = np.random.uniform(size=[1])
        return state, terminal, reward

    @property
    def states(self):
        return dict(shape=1, type='float')

    @property
    def actions(self):
        return dict(type='float', min_value=0.0, max_value=1.0)


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


def create_agent(environment, network_spec, saver_spec=None):
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
        network=network_spec,
        saver=saver_spec
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

        runner = Runner(agent=agent, environment=environment)
        runner.run(episodes=100)
        runner.close()

    def test_auto_save_restore(self):
        saver_steps = 15
        steps_per_episode = 20
        train_episodes = 2

        assert ((steps_per_episode + 1) * train_episodes % saver_steps) > 0

        environment = DummyEnv()
        network_spec = [
            dict(type='dense', size=4)
        ]
        model_path = self._tmp_dir_path + "/model_auto_save"

        saver_spec = dict(
            directory=model_path,
            steps=saver_steps,
            load=False
        )
        agent = create_agent(environment, network_spec, saver_spec)
        runner = Runner(agent=agent, environment=environment)

        runner.run(max_episode_timesteps=steps_per_episode, episodes=train_episodes)
        # Deliberately avoid closing the runner/agent to simulate unexpected shutdown

        saver_spec["load"] = True
        agent = create_agent(environment, network_spec, saver_spec)
        expected_timestep = train_episodes * (steps_per_episode + 1) // saver_steps * saver_steps
        assert agent.episode == train_episodes - 1
        assert agent.timestep == expected_timestep

        runner = Runner(agent=agent, environment=environment)
        runner.run(max_episode_timesteps=steps_per_episode, episodes=train_episodes)
        assert agent.episode == 2 * train_episodes - 1
        runner.close()

    def test_restore_from_checkpoint(self):
        saver_steps = 15
        steps_per_episode = 20
        train_episodes = 2

        assert ((steps_per_episode + 1) * train_episodes % saver_steps) > 0

        environment = DummyEnv()
        network_spec = [
            dict(type='dense', size=4)
        ]
        model_path = self._tmp_dir_path + "/model_auto_save"

        saver_spec = dict(
            directory=model_path,
            steps=saver_steps,
            load=False
        )
        agent = create_agent(environment, network_spec, saver_spec)
        runner = Runner(agent=agent, environment=environment)

        runner.run(max_episode_timesteps=steps_per_episode, episodes=train_episodes)
        # Deliberately avoid closing the runner/agent to simulate unexpected shutdown

        agent = create_agent(environment, network_spec)
        agent.restore_model(directory=model_path)
        agent.reset()
        expected_timestep = train_episodes * (steps_per_episode + 1) // saver_steps * saver_steps
        assert agent.episode == train_episodes - 1
        assert agent.timestep == expected_timestep

        runner = Runner(agent=agent, environment=environment)
        runner.run(max_episode_timesteps=steps_per_episode, episodes=train_episodes)
        assert agent.episode == 2 * train_episodes - 1
        runner.close()

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
