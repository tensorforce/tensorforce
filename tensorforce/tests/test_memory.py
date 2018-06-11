import unittest

import tensorflow as tf
import numpy as np
from tensorforce.core.memories import Queue, Latest


class TestMemory(unittest.TestCase):
    states_spec = dict(
        test_state=dict(
            shape=[1],
            type="float"
        )
    )
    actions_spec = dict(
        test_action=dict(
            shape=[],
            type="float"
        )
    )
    internals_spec = dict()

    def setUp(self):
        self.states = dict(
            test_state=tf.placeholder(dtype=tf.float32, shape=[None, 1], )
        )
        self.actions = dict(
            test_action=tf.placeholder(dtype=tf.float32, shape=[None], )
        )
        self.terminal = tf.placeholder(dtype=tf.bool, shape=[None])
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None])

        def custom_getter(getter, name, registered=False, **kwargs):
            return getter(name=name, **kwargs)

        self.variable_scope = tf.variable_scope("test_memory", custom_getter=custom_getter)
        self.variable_scope.__enter__()

        self.sess = tf.Session()

    def tearDown(self):
        self.variable_scope.__exit__(None, None, None)
        self.sess.close()
        tf.reset_default_graph()

    def _build_store_op(self, mem):
        return mem.store(
            states=self.states,
            internals=dict(),
            actions=self.actions,
            terminal=self.terminal,
            reward=self.reward
        )

    def test_queue(self):
        episode_length = 2
        capacity_episodes = 5

        mem = Queue(
            states=self.states_spec,
            internals=self.internals_spec,
            actions=self.actions_spec,
            include_next_states=False,
            capacity=episode_length * capacity_episodes
        )

        mem.initialize()
        store_op = self._build_store_op(mem)

        self.sess.run(tf.global_variables_initializer())

        for i in range(capacity_episodes):
            self.sess.run(store_op, feed_dict={
                self.states["test_state"]: np.ones(shape=[episode_length, 1]) * i,
                self.actions["test_action"]: np.random.uniform(size=[episode_length]),
                self.terminal: np.array([False] * (episode_length - 1) + [True]),
                self.reward: np.random.uniform(size=[episode_length])
            })

        episodes_inserted = self.sess.run(mem.episode_count)
        assert episodes_inserted == capacity_episodes

    def test_latest(self):
        episode_length = 2
        capacity_episodes = 2
        mem = Latest(
            states=self.states_spec,
            internals=self.internals_spec,
            actions=self.actions_spec,
            include_next_states=False,
            capacity=episode_length * capacity_episodes + 1
        )

        n = tf.placeholder(dtype=tf.int32)

        mem.initialize()
        store_op = self._build_store_op(mem)
        retrieve_op = mem.retrieve_episodes(n)

        self.sess.run(tf.global_variables_initializer())

        for i in range(capacity_episodes + 2):
            self.sess.run(store_op, feed_dict={
                self.states["test_state"]: np.ones(shape=[episode_length, 1]) * i,
                self.actions["test_action"]: np.random.uniform(size=[episode_length]),
                self.terminal: np.array([False] * (episode_length - 1) + [True]),
                self.reward: np.random.uniform(size=[episode_length])
            })

        retrieved_data = self.sess.run(retrieve_op, feed_dict={n: capacity_episodes})
        assert np.alen(retrieved_data["terminal"]) > 0


if __name__ == "__main__":
    unittest.main()
