import unittest

import tensorflow as tf
import numpy as np
from tensorforce.core.memories import Queue, Latest, Replay


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
        self._episode_index = 0
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

    def _make_mem(self, clazz, capacity, include_next_states=False):
        return clazz(
            states=self.states_spec,
            internals=self.internals_spec,
            actions=self.actions_spec,
            include_next_states=include_next_states,
            capacity=capacity
        )

    def _store_episode(self, store_op, episode_length):
        self.sess.run(store_op, feed_dict={
            self.states["test_state"]: np.ones(shape=[episode_length, 1]) * self._episode_index,
            self.actions["test_action"]: np.random.uniform(size=[episode_length]),
            self.terminal: np.array([False] * (episode_length - 1) + [True]),
            self.reward: np.random.uniform(size=[episode_length])
        })
        self._episode_index += 1

    def test_queue(self):
        episode_length = 3
        capacity_episodes = 2

        mem = self._make_mem(Queue, episode_length * capacity_episodes)
        mem.initialize()
        store_op = self._build_store_op(mem)

        self.sess.run(tf.global_variables_initializer())

        for i in range(capacity_episodes + 1):
            self._store_episode(store_op=store_op, episode_length=episode_length)
            episodes_inserted = self.sess.run(mem.episode_count)
            assert min(capacity_episodes, i + 1) == episodes_inserted

        episodes_inserted = self.sess.run(mem.episode_count)
        assert capacity_episodes == episodes_inserted

    def test_queue_not_aligned(self):
        episode_length = 3
        num_full_episodes = 2

        mem = self._make_mem(Queue, episode_length * (num_full_episodes + 1) - 1)
        true_capacity_episodes = num_full_episodes + 1

        mem.initialize()
        store_op = self._build_store_op(mem)

        self.sess.run(tf.global_variables_initializer())

        for i in range(true_capacity_episodes + 1):
            self._store_episode(store_op=store_op, episode_length=episode_length)
            episodes_inserted = self.sess.run(mem.episode_count)
            assert min(true_capacity_episodes, i + 1) == episodes_inserted

        episodes_inserted = self.sess.run(mem.episode_count)
        assert true_capacity_episodes == episodes_inserted

    def test_latest_episodes(self):
        episode_length = 2
        num_full_episodes = 2
        capacity = episode_length * num_full_episodes + 1
        mem = self._make_mem(Latest, capacity=capacity)
        true_capacity_episodes = num_full_episodes + 1

        n = tf.placeholder(dtype=tf.int32)

        mem.initialize()
        store_op = self._build_store_op(mem)
        retrieve_op_e = mem.retrieve_episodes(n)

        self.sess.run(tf.global_variables_initializer())

        for _ in range(num_full_episodes):
            self._store_episode(store_op=store_op, episode_length=episode_length)
        episodes_inserted = self.sess.run(mem.episode_count)
        assert num_full_episodes == episodes_inserted

        retrieved_data = self.sess.run(retrieve_op_e, feed_dict={n: num_full_episodes})
        assert [False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 0.0, 1.0, 1.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_e, feed_dict={n: num_full_episodes + 1})
        assert [False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 0.0, 1.0, 1.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_e, feed_dict={n: num_full_episodes - 1})
        assert [False, True] == retrieved_data["terminal"].tolist()
        assert [1.0, 1.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        self._store_episode(store_op=store_op, episode_length=episode_length)
        episodes_inserted = self.sess.run(mem.episode_count)
        assert true_capacity_episodes == episodes_inserted

        retrieved_data = self.sess.run(retrieve_op_e, feed_dict={n: num_full_episodes})
        assert [False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [1.0, 1.0, 2.0, 2.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_e, feed_dict={n: num_full_episodes + 1})
        assert [True, False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 1.0, 1.0, 2.0, 2.0] == retrieved_data["states"]["test_state"].flatten().tolist()

    def test_latest_timesteps(self):
        episode_length = 2
        num_full_episodes = 2
        capacity = episode_length * num_full_episodes + 1
        mem = self._make_mem(Latest, capacity=capacity)

        n = tf.placeholder(dtype=tf.int32)

        mem.initialize()
        store_op = self._build_store_op(mem)
        retrieve_op_t = mem.retrieve_timesteps(n)

        self.sess.run(tf.global_variables_initializer())

        for _ in range(num_full_episodes):
            self._store_episode(store_op=store_op, episode_length=episode_length)
        episodes_inserted = self.sess.run(mem.episode_count)
        assert num_full_episodes == episodes_inserted

        retrieved_data = self.sess.run(retrieve_op_t, feed_dict={n: num_full_episodes * episode_length})
        assert [False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 0.0, 1.0, 1.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_t, feed_dict={n: capacity})
        assert [False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 0.0, 1.0, 1.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_t, feed_dict={n: capacity + 1})
        assert [False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 0.0, 1.0, 1.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_t, feed_dict={n: num_full_episodes * episode_length - 1})
        assert [True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 1.0, 1.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_t, feed_dict={n: 1})
        assert [True] == retrieved_data["terminal"].tolist()
        assert [1.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_t, feed_dict={n: 0})
        assert [] == retrieved_data["terminal"].tolist()
        assert [] == retrieved_data["states"]["test_state"].flatten().tolist()

        self._store_episode(store_op=store_op, episode_length=episode_length)

        retrieved_data = self.sess.run(retrieve_op_t, feed_dict={n: num_full_episodes * episode_length})
        assert [False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [1.0, 1.0, 2.0, 2.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_t, feed_dict={n: capacity})
        assert [True, False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 1.0, 1.0, 2.0, 2.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        retrieved_data = self.sess.run(retrieve_op_t, feed_dict={n: capacity + 1})
        assert [True, False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 1.0, 1.0, 2.0, 2.0] == retrieved_data["states"]["test_state"].flatten().tolist()

    @unittest.skip(reason="https://github.com/reinforceio/tensorforce/issues/128")
    def test_latest_sequences(self):
        episode_length = 3
        seq_length = 2
        num_full_episodes = 2
        capacity = episode_length * num_full_episodes + 1
        mem = self._make_mem(Latest, capacity=capacity)

        n = tf.placeholder(dtype=tf.int32)

        mem.initialize()
        store_op = self._build_store_op(mem)
        retrieve_op_seq = mem.retrieve_sequences(n, seq_length)

        self.sess.run(tf.global_variables_initializer())

        for _ in range(num_full_episodes):
            self._store_episode(store_op=store_op, episode_length=episode_length)
        episodes_inserted = self.sess.run(mem.episode_count)
        assert num_full_episodes == episodes_inserted

        retrieved_data = self.sess.run(retrieve_op_seq, feed_dict={n: num_full_episodes * episode_length})
        print(retrieved_data)
        assert [False, True, False, True] == retrieved_data["terminal"].tolist()
        assert [0.0, 0.0, 1.0, 1.0] == retrieved_data["states"]["test_state"].flatten().tolist()

        self._store_episode(store_op=store_op, episode_length=episode_length)

    def test_replay_episodes(self):
        episode_length = 2
        num_full_episodes = 2
        capacity = episode_length * num_full_episodes + 1
        mem = self._make_mem(Replay, capacity=capacity)
        true_capacity_episodes = num_full_episodes + 1

        mem.initialize()
        store_op = self._build_store_op(mem)
        retrieve_op_full = mem.retrieve_episodes(num_full_episodes)
        retrieve_op_full_plus = mem.retrieve_episodes(num_full_episodes + 1)
        retrieve_op_full_minus = mem.retrieve_episodes(num_full_episodes - 1)

        self.sess.run(tf.global_variables_initializer())

        try:
            self.sess.run(retrieve_op_full)
            assert False
        except tf.errors.InvalidArgumentError as e:
            assert "nothing stored yet" in e.message

        for _ in range(num_full_episodes):
            self._store_episode(store_op=store_op, episode_length=episode_length)
        episodes_inserted = self.sess.run(mem.episode_count)
        assert num_full_episodes == episodes_inserted

        retrieved_data = self.sess.run(retrieve_op_full)
        assert [False, True, False, True] == retrieved_data["terminal"].tolist()
        states = retrieved_data["states"]["test_state"].flatten().tolist()
        for i in range(0, len(states), episode_length):
            assert states[i] == states[i + 1]

        retrieved_data = self.sess.run(retrieve_op_full_plus)
        assert [False, True, False, True, False, True] == retrieved_data["terminal"].tolist()
        states = retrieved_data["states"]["test_state"].flatten().tolist()
        for i in range(0, len(states), episode_length):
            assert states[i] == states[i + 1]

        retrieved_data = self.sess.run(retrieve_op_full_minus)
        assert [False, True] == retrieved_data["terminal"].tolist()
        states = retrieved_data["states"]["test_state"].flatten().tolist()
        for i in range(0, len(states), episode_length):
            assert states[i] == states[i + 1]

        self._store_episode(store_op=store_op, episode_length=episode_length)
        episodes_inserted = self.sess.run(mem.episode_count)
        assert true_capacity_episodes == episodes_inserted

        retrieved_data = self.sess.run(retrieve_op_full)
        # We avoid explicit check as the 0-th episode is partial and can be selected
        assert np.sum(retrieved_data["terminal"]) == num_full_episodes

        retrieved_data = self.sess.run(retrieve_op_full_plus)
        assert np.sum(retrieved_data["terminal"]) == true_capacity_episodes


if __name__ == "__main__":
    unittest.main()
