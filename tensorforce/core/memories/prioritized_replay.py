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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorforce import util
from tensorforce.core.memories import Queue


class PrioritizedReplay(Queue):
    """
    Naive prioritized replay in TensorFlow.
    """

    def __init__(
        self,
        states,
        internals,
        actions,
        include_next_states,
        capacity,
        prioritization_weight=1.0,
        scope='queue',
        summary_labels=None
    ):
        super(PrioritizedReplay, self).__init__(
            states=states,
            internals=internals,
            actions=actions,
            include_next_states=include_next_states,
            capacity=capacity,
            scope=scope,
            summary_labels=summary_labels
        )
        self.prioritization_weight = prioritization_weight

    def tf_initialize(self):
        super(PrioritizedReplay, self).tf_initialize()

        # Priorities
        self.priorities = tf.get_variable(
            name='priorities',
            shape=(self.capacity,),
            dtype=util.tf_dtype('float'),
            trainable=False
        )

        # Index of experience which have not been evaluated yet.
        self.no_priority_index = tf.get_variable(
            name='no-priority-index',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )

        # Indices of batches
        self.batch_indices = tf.get_variable(
            name='batch-indices',
            dtype=util.tf_dtype('int'),
            shape=(self.capacity,),
            initializer=0,
            trainable=False
        )

    def tf_store(self, states, internals, actions, terminal, reward):
        # Memory indices to overwrite.
        num_instances = tf.shape(input=terminal)[0]
        indices = tf.range(start=self.memory_index, limit=(self.memory_index + num_instances)) % self.capacity

        # Remove episode indices.
        num_episodes = tf.count_nonzero(
            input_tensor=tf.gather(params=self.terminal_memory, indices=indices),
            axis=0,
            dtype=util.tf_dtype('int')
        )
        num_episodes = tf.minimum(x=num_episodes, y=self.episode_count)
        assignment = tf.assign(
            ref=self.episode_indices[:self.episode_count + 1 - num_episodes],
            value=self.episode_indices[num_episodes: self.episode_count + 1]
        )

        # Decrement episode count.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = tf.assign_sub(ref=self.episode_count, value=num_episodes)

        # Update no-priority index.
        # TODO check if correct.
        with tf.control_dependencies(control_inputs=(assignment,)):
            update_no_priority = tf.cond(
                pred=(self.memory_index < self.capacity),
                true_fn=(lambda: -num_instances),
                false_fn=(lambda: 0)
            )
            assignment = tf.assign_sub(
                ref=self.no_priority_index,
                value=update_no_priority
            )

        # Assign new observations.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignments = list()
            for name, state in states.items():
                assignments.append(tf.scatter_update(ref=self.states_memory[name], indices=indices, updates=state))
            for name, internal in internals.items():
                assignments.append(tf.scatter_update(
                    ref=self.internals_memory[name],
                    indices=indices,
                    updates=internal
                ))
            for name, action in actions.items():
                assignments.append(tf.scatter_update(ref=self.actions_memory[name], indices=indices, updates=action))
            assignments.append(tf.scatter_update(ref=self.terminal_memory, indices=indices, updates=terminal))
            assignments.append(tf.scatter_update(ref=self.reward_memory, indices=indices, updates=reward))

        # Increment memory index.
        with tf.control_dependencies(control_inputs=assignments):
            assignment = tf.assign(ref=self.memory_index, value=((self.memory_index + num_instances) % self.capacity))

        # Add episode indices.
        with tf.control_dependencies(control_inputs=(assignment,)):
            num_episodes = tf.count_nonzero(input_tensor=terminal, axis=0, dtype=util.tf_dtype('int'))
            assignment = tf.assign(
                ref=self.episode_indices[self.episode_count + 1: self.episode_count + 1 + num_episodes],
                value=tf.boolean_mask(tensor=indices, mask=terminal)
            )

        # Increment episode count.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = tf.assign_add(ref=self.episode_count, value=num_episodes)

        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf.no_op()

    def tf_retrieve_timesteps(self, n):
        num_timesteps = (self.memory_index - self.episode_indices[0] - 2) % self.capacity + 1
        n = tf.minimum(x=n, y=num_timesteps)

        indices = tf.zeros(shape=(n,), dtype=tf.int32)
        # Only compute once outside loop.
        sum_priorities = tf.reduce_sum(input_tensor=self.priorities, axis=0)
        loop_index = tf.get_variable(
            name='loop-index',
            dtype=tf.int32,
            initializer=0,
            trainable=False
        )

        def cond(sum_priorities, priorities, indices, loop_index, n):
            return tf.less(loop_index, n)

        def sampling_fn(sum_priorities, priorities, indices, loop_index, n):
            # 3 cases:
            # 1. not_sampled_index < len(self.observations):
            # 2. sum_priorities / self.capacity < util.epsilon:
            # 3. all other cases

            def true_fn():
                # tf cond on
                # 1. not_sampled_index < len(self.observations):
                # -> fetch not sampled index
                # 2. sum_priorities / self.capacity < util.epsilon:
                # -> randomly sample
                pass

            def false_fn():
                # Priority sampling loop.
                pass

            index = tf.cond(
                pred=tf.logical_or(x=(self.no_priority_index < n), y=(sum_priorities / self.capacity < util.epsilon)),
                true_fn=true_fn,
                false_fn=false_fn
            )

        tf.while_loop(
            cond=cond,
            body=sampling_fn,
            loop_vars=[sum_priorities, self.priorities, indices, loop_index, num_timesteps]
        )

        # Save batch indices.
        assignment = tf.assign(ref=self.batch_indices, value=indices)

        with tf.control_dependencies(control_inputs=(assignment,)):
            terminal = tf.gather(params=self.terminal_memory, indices=indices)
            indices = tf.boolean_mask(tensor=indices, mask=tf.logical_not(x=terminal))

        return self.retrieve_indices(indices=indices)

    def tf_update_batch(self, loss_per_instance):
        # TODO
        # 1. Compute priority based on losses and matching batch indices
        # 2. Compute priority order?
        pass

        # if self.batch_indices is None:
        #     raise TensorForceError("Need to call get_batch before each update_batch call.")
        #     # if len(loss_per_instance) != len(self.batch_indices):
        #     #     raise TensorForceError("For all instances a loss value has to be provided.")
        #
        # updated = list()
        # for index, loss in zip(self.batch_indices, loss_per_instance):
        #     priority, observation = self.observations[index]
        #     updated.append((loss ** self.prioritization_weight, observation))
        # for index in sorted(self.batch_indices, reverse=True):
        #     priority, _ = self.observations.pop(index)
        #     self.none_priority_index -= (priority is not None)
        # self.batch_indices = None
        # updated = sorted(updated, key=(lambda x: x[0]))
        #
        # update_priority, update_observation = updated.pop()
        # index = -1
        # for priority, _ in iter(self.observations):
        #     index += 1
        #     if index == self.none_priority_index:
        #         break
        #     if update_priority < priority:
        #         continue
        #     self.observations.insert(index, (update_priority, update_observation))
        #     index += 1
        #     self.none_priority_index += 1
        #     if not updated:
        #         break
        #     update_priority, update_observation = updated.pop()
        # else:
        #     self.observations.insert(index, (update_priority, update_observation))
        #     self.none_priority_index += 1
        # while updated:
        #     self.observations.insert(index, updated.pop())
        #     self.none_priority_index += 1
