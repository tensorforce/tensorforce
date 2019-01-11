# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

import tensorflow as tf

from tensorforce import util
from tensorforce.core.memories import Memory


class PrioritizedReplay(Memory):
    """
    Memory organized as a priority queue, which randomly retrieves experiences sampled according
    their priority values.
    """

    def __init__(
        self,
        name,
        states_spec,
        internals_spec,
        actions_spec,
        include_next_state,
        capacity,
        prioritization_weight=1.0,
        buffer_size=100
    ):
        """
        Prioritized experience replay.

        Args:
            states: States specification.
            internals: Internal states specification.
            actions: Actions specification.
            include_next_states: Include subsequent state if true.
            capacity: Memory capacity.
            prioritization_weight: Prioritization weight.
            buffer_size: Buffer size. The buffer is used to insert experiences before experiences
                have been computed via updates. Note that if the buffer is to small in comparison
                to the frequency with which updates are performed, old experiences from the buffer
                will be overwritten before they are moved to the main memory.
        """
        super().__init__(
            name=name, states_spec=states_spec, internals_spec=internals_spec,
            actions_spec=actions_spec, include_next_state=include_next_state
        )

        self.capacity = capacity
        self.buffer_size = buffer_size
        self.prioritization_weight = prioritization_weight

        self.retrieve_indices = None

        self.states_memory = dict()
        self.internals_memory = dict()
        self.actions_memory = dict()
        self.terminal_memory = None
        self.reward_memory = None
        self.memory_index = None
        self.priorities = None

        self.buffer_index = None
        self.states_buffer = dict()
        self.internals_buffer = dict()
        self.actions_buffer = dict()
        self.terminal_buffer = None
        self.reward_buffer = None
        self.batch_indices = None
        self.last_batch_buffer_elems = None
        self.memory_size = None

    def tf_initialize(self):
        super().tf_initialize()

        # States
        for name in sorted(self.states_spec):
            state = self.states_spec[name]
            self.states_memory[name] = self.add_variable(
                name=('state-' + name),
                dtype=state['type'],
                shape=(self.capacity,) + tuple(state['shape']),
                trainable=False
            )

        # Internals
        for name in sorted(self.internals_spec):
            internal = self.internals_spec[name]
            self.internals_memory[name] = self.add_variable(
                name=('internal-' + name),
                dtype=internal['type'],
                shape=(self.capacity,) + tuple(internal['shape']),
                trainable=False
            )

        # Actions
        for name in sorted(self.actions_spec):
            action = self.actions_spec[name]
            self.actions_memory[name] = self.add_variable(
                name=('action-' + name),
                dtype=action['type'],
                shape=(self.capacity,) + tuple(action['shape']),
                trainable=False
            )

        # Terminal
        self.terminal_memory = self.add_variable(
            name='terminal',
            dtype='bool',
            shape=(self.capacity,),
            trainable=False
        )

        # Reward
        self.reward_memory = self.add_variable(
            name='reward',
            dtype='float',
            shape=(self.capacity,),
            trainable=False
        )

        # Memory index - current insertion index.
        self.memory_index = self.add_variable(
            name='memory-index',
            dtype='int',
            shape=(),
            trainable=False,
            initializer='zeros'
        )

        # Priorities
        self.priorities = self.add_variable(
            name='priorities',
            dtype='float',
            shape=(self.capacity,),
            trainable=False
        )

        # Buffer variables. The buffer is used to insert data for which we
        # do not have priorities yet.
        self.buffer_index = self.add_variable(
            name='buffer-index',
            dtype='int',
            shape=(),
            trainable=False,
            initializer='zeros'
        )

        # States
        for name in sorted(self.states_spec):
            state = self.states_spec[name]
            self.states_buffer[name] = self.add_variable(
                name=('state-buffer-' + name),
                dtype=state['type'],
                shape=(self.buffer_size,) + tuple(state['shape']),
                trainable=False
            )

        # Internals
        for name in sorted(self.internals_spec):
            internal = self.internals_spec[name]
            self.internals_buffer[name] = self.add_variable(
                name=('internal-buffer-' + name),
                dtype=internal['type'],
                shape=(self.capacity,) + tuple(internal['shape']),
                trainable=False
            )

        # Actions
        for name in sorted(self.actions_spec):
            action = self.actions_spec[name]
            self.actions_buffer[name] = self.add_variable(
                name=('action-buffer-' + name),
                dtype=action['type'],
                shape=(self.buffer_size,) + tuple(action['shape']),
                trainable=False
            )

        # Terminal
        self.terminal_buffer = self.add_variable(
            name='terminal-buffer',
            dtype='bool',
            shape=(self.capacity,),
            initializer=tf.constant_initializer(
                value=tuple(n == self.buffer_size - 1 for n in range(self.capacity)),
                dtype=util.tf_dtype('bool')
            ),
            trainable=False
        )

        # Reward
        self.reward_buffer = self.add_variable(
            name='reward-buffer',
            dtype='float',
            shape=(self.buffer_size,),
            trainable=False
        )

        # Indices of batch experiences in main memory.
        self.batch_indices = self.add_variable(
            name='batch-indices',
            dtype='int',
            shape=(self.capacity,),
            trainable=False
        )

        # Number of elements taken from the buffer in the last batch.
        self.last_batch_buffer_elems = self.add_variable(
            name='last-batch-buffer-elems',
            dtype='int',
            shape=(),
            trainable=False,
            initializer='zeros'
        )
        self.memory_size = self.add_variable(
            name='memory-size',
            dtype='int',
            shape=(),
            trainable=False,
            initializer='zeros'
        )

    def tf_store(self, states, internals, actions, terminal, reward):
        # We first store new experiences into a buffer that is separate from main memory.
        # We insert these into the main memory once we have computed priorities on a given batch.
        num_instances = tf.shape(input=terminal)[0]

        # Simple way to prevent buffer overflows.
        start_index = self.cond(
            # Why + 1? Because of next state, otherwise that has to be handled separately.
            pred=(self.buffer_index + num_instances + 1 >= self.buffer_size),
            true_fn=(lambda: 0),
            false_fn=(lambda: self.buffer_index)
        )
        end_index = start_index + num_instances

        # Assign new observations.
        assignments = list()
        for name in sorted(states):
            assignments.append(tf.assign(ref=self.states_buffer[name][start_index:end_index], value=states[name]))
        for name in sorted(internals):
            assignments.append(tf.assign(
                ref=self.internals_buffer[name][start_index:end_index],
                value=internals[name]
            ))
        for name in sorted(actions):
            assignments.append(tf.assign(ref=self.actions_buffer[name][start_index:end_index], value=actions[name]))

        assignments.append(tf.assign(ref=self.terminal_buffer[start_index:end_index], value=terminal))
        assignments.append(tf.assign(ref=self.reward_buffer[start_index:end_index], value=reward))

        # Increment memory index.
        with tf.control_dependencies(control_inputs=assignments):
            assignment = tf.assign(ref=self.buffer_index, value=(start_index + num_instances))

        with tf.control_dependencies(control_inputs=(assignment,)):
            return util.no_operation()

    def tf_retrieve_timesteps(self, n):
        num_buffer_elems = tf.minimum(x=self.buffer_index, y=n)

        # We can only sample from priority memory if buffer elements were previously inserted.
        num_priority_elements = self.cond(
            pred=self.memory_size > 0,
            true_fn=lambda: n - num_buffer_elems,
            false_fn=lambda: 0
        )

        def sampling_fn():
            # Vectorized sampling.
            sum_priorities = tf.reduce_sum(input_tensor=self.priorities, axis=0)
            sample = tf.random_uniform(shape=(num_priority_elements,), dtype=tf.float32)
            indices = tf.zeros(shape=(num_priority_elements,), dtype=tf.int32)

            def cond(loop_index, sample):
                return tf.reduce_all(input_tensor=(sample <= 0.0))

            def sampling_body(loop_index, sample):
                priority = tf.gather(params=self.priorities, indices=loop_index)
                sample -= priority / sum_priorities
                loop_index += tf.cast(
                    x=(sample > 0.0),
                    dtype=tf.int32,
                )

                return loop_index, sample

            priority_indices = self.while_loop(
                cond=cond,
                body=sampling_body,
                loop_vars=(indices, sample)
            )[0]
            return priority_indices

        # Reset batch indices.
        assignment = tf.assign(
            ref=self.batch_indices,
            value=tf.zeros(shape=tf.shape(self.batch_indices), dtype=tf.int32)
        )
        with tf.control_dependencies(control_inputs=(assignment,)):
            priority_indices = self.cond(
                pred=num_priority_elements > 0,
                true_fn=sampling_fn,
                false_fn=lambda: tf.zeros(shape=(num_priority_elements,), dtype=tf.int32)
            )
            priority_terminal = tf.gather(params=self.terminal_memory, indices=priority_indices)
            priority_indices = tf.boolean_mask(tensor=priority_indices, mask=tf.logical_not(x=priority_terminal))

            # Store how many elements we retrieved from the buffer for updating priorities.
            # Note that this is just the count, as we can reconstruct the indices from that.
            assignments = list()
            assignments.append(tf.assign(ref=self.last_batch_buffer_elems, value=num_buffer_elems))

            # Store indices used from priority memory. Note that these are the full indices
            # as they were not taken in order.
            update = tf.ones(shape=tf.shape(input=priority_indices), dtype=tf.int32)
            assignments.append(tf.scatter_update(
                ref=self.batch_indices,
                indices=priority_indices,
                updates=update
            ))
        # Fetch results.
        with tf.control_dependencies(control_inputs=assignments):
            return self.retrieve_indices(buffer_elements=num_buffer_elems, priority_indices=priority_indices)

    def tf_retrieve_indices(self, buffer_elements, priority_indices):
        """
        Fetches experiences for given indices by combining entries from buffer
        which have no priorities, and entries from priority memory.

        Args:
            buffer_elements: Number of buffer elements to retrieve
            priority_indices: Index tensor for priority memory

        Returns: Batch of experiences
        """
        states = dict()
        buffer_start = self.buffer_index - buffer_elements
        buffer_end = self.buffer_index

        # Fetch entries from respective memories, concat.
        for name in sorted(self.states_memory):
            buffer_state_memory = self.states_buffer[name]
            # Slicing is more efficient than gathering, and buffer elements are always
            # fetched using contiguous indices.
            buffer_states = buffer_state_memory[buffer_start:buffer_end]
            # Memory indices are obtained via priority sampling, hence require gather.
            memory_states = tf.gather(params=self.states_memory[name], indices=priority_indices)
            states[name] = tf.concat(values=(buffer_states, memory_states), axis=0)

        internals = dict()
        for name in sorted(self.internals_memory):
            internal_buffer_memory = self.internals_buffer[name]
            buffer_internals = internal_buffer_memory[buffer_start:buffer_end]
            memory_internals = tf.gather(params=self.internals_memory[name], indices=priority_indices)
            internals[name] = tf.concat(values=(buffer_internals, memory_internals), axis=0)

        actions = dict()
        for name in sorted(self.actions_memory):
            action_buffer_memory = self.actions_buffer[name]
            buffer_action = action_buffer_memory[buffer_start:buffer_end]
            memory_action = tf.gather(params=self.actions_memory[name], indices=priority_indices)
            actions[name] = tf.concat(values=(buffer_action, memory_action), axis=0)

        buffer_terminal = self.terminal_buffer[buffer_start:buffer_end]
        priority_terminal = tf.gather(params=self.terminal_memory, indices=priority_indices)
        terminal = tf.concat(values=(buffer_terminal, priority_terminal), axis=0)

        buffer_reward = self.reward_buffer[buffer_start:buffer_end]
        priority_reward = tf.gather(params=self.reward_memory, indices=priority_indices)
        reward = tf.concat(values=(buffer_reward, priority_reward), axis=0)

        if self.include_next_states:
            assert util.rank(priority_indices) == 1
            next_priority_indices = (priority_indices + 1) % self.capacity
            next_buffer_start = (buffer_start + 1) % self.buffer_size
            next_buffer_end = (buffer_end + 1) % self.buffer_size

            next_states = dict()
            for name in sorted(self.states_memory):
                buffer_state_memory = self.states_buffer[name]
                buffer_next_states = buffer_state_memory[next_buffer_start:next_buffer_end]
                memory_next_states = tf.gather(params=self.states_memory[name], indices=next_priority_indices)
                next_states[name] = tf.concat(values=(buffer_next_states, memory_next_states), axis=0)

            next_internals = dict()
            for name in sorted(self.internals_memory):
                buffer_internal_memory = self.internals_buffer[name]
                buffer_next_internals = buffer_internal_memory[next_buffer_start:next_buffer_end]
                memory_next_internals = tf.gather(params=self.internals_memory[name], indices=next_priority_indices)
                next_internals[name] = tf.concat(values=(buffer_next_internals, memory_next_internals), axis=0)

            return dict(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward,
                next_states=next_states,
                next_internals=next_internals
            )
        else:
            return dict(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward
            )

    def tf_update_batch(self, loss_per_instance):
        """
        Updates priority memory by performing the following steps:

        1. Use saved indices from prior retrieval to reconstruct the batch
        elements which will have their priorities updated.
        2. Compute priorities for these elements.
        3. Insert buffer elements to memory, potentially overwriting existing elements.
        4. Update priorities of existing memory elements
        5. Resort memory.
        6. Update buffer insertion index.

        Note that this implementation could be made more efficient by maintaining
        a sorted version via sum trees.

        :param loss_per_instance: Losses from recent batch to perform priority update
        """
        # 1. We reconstruct the batch from the buffer and the priority memory via
        # the TensorFlow variables holding the respective indices.
        mask = tf.not_equal(
            x=self.batch_indices,
            y=tf.zeros(shape=tf.shape(input=self.batch_indices), dtype=tf.int32)
        )
        priority_indices = tf.reshape(tensor=tf.where(condition=mask), shape=[-1])

        # These are elements from the buffer which first need to be inserted into the main memory.
        sampled_buffer_batch = self.tf_retrieve_indices(
            buffer_elements=self.last_batch_buffer_elems,
            priority_indices=priority_indices
        )

        # Extract batch elements.
        states = sampled_buffer_batch['states']
        internals = sampled_buffer_batch['internals']
        actions = sampled_buffer_batch['actions']
        terminal = sampled_buffer_batch['terminal']
        reward = sampled_buffer_batch['reward']

        # 2. Compute priorities for all batch elements.
        priorities = loss_per_instance ** self.prioritization_weight
        assignments = list()

        # 3. Insert the buffer elements from the recent batch into memory,
        # overwrite memory if full.
        memory_end_index = self.memory_index + self.last_batch_buffer_elems
        memory_insert_indices = tf.range(
            start=self.memory_index,
            limit=memory_end_index
        ) % self.capacity

        for name in sorted(states):
            assignments.append(tf.scatter_update(
                ref=self.states_memory[name],
                indices=memory_insert_indices,
                # Only buffer elements from batch.
                updates=states[name][0:self.last_batch_buffer_elems])
            )
        for name in sorted(internals):
            assignments.append(tf.scatter_update(
                ref=self.internals_buffer[name],
                indices=memory_insert_indices,
                updates=internals[name][0:self.last_batch_buffer_elems]
            ))
        assignments.append(tf.scatter_update(
            ref=self.priorities,
            indices=memory_insert_indices,
            updates=priorities[0:self.last_batch_buffer_elems]
        ))
        assignments.append(tf.scatter_update(
            ref=self.terminal_memory,
            indices=memory_insert_indices,
            updates=terminal[0:self.last_batch_buffer_elems])
        )
        assignments.append(tf.scatter_update(
            ref=self.reward_memory,
            indices=memory_insert_indices,
            updates=reward[0:self.last_batch_buffer_elems])
        )
        for name in sorted(actions):
            assignments.append(tf.scatter_update(
                ref=self.actions_memory[name],
                indices=memory_insert_indices,
                updates=actions[name][0:self.last_batch_buffer_elems]
            ))

        # 4.Update the priorities of the elements already in the memory.
        # Slice out remaining elements - [] if all batch elements were from buffer.
        main_memory_priorities = priorities[self.last_batch_buffer_elems:]
        # Note that priority indices can have a different shape because multiple
        # samples can be from the same index.
        main_memory_priorities = main_memory_priorities[0:tf.shape(priority_indices)[0]]
        assignments.append(tf.scatter_update(
            ref=self.priorities,
            indices=priority_indices,
            updates=main_memory_priorities
        ))

        with tf.control_dependencies(control_inputs=assignments):
            # 5. Re-sort memory according to priorities.
            assignments = list()

            # Obtain sorted order and indices.
            sorted_priorities, sorted_indices = tf.nn.top_k(
                input=self.priorities,
                k=self.capacity,
                sorted=True
            )
            # Re-assign elements according to priorities.
            # Priorities was the tensor we used to sort, so this can be directly assigned.
            assignments.append(tf.assign(ref=self.priorities, value=sorted_priorities))

            # All other memory variables are assigned via scatter updates using the indices
            # returned by the sort:
            assignments.append(tf.scatter_update(
                ref=self.terminal_memory,
                indices=sorted_indices,
                updates=self.terminal_memory
            ))
            for name in sorted(self.states_memory):
                assignments.append(tf.scatter_update(
                    ref=self.states_memory[name],
                    indices=sorted_indices,
                    updates=self.states_memory[name]
                ))
            for name in sorted(self.actions_memory):
                assignments.append(tf.scatter_update(
                    ref=self.actions_memory[name],
                    indices=sorted_indices,
                    updates=self.actions_memory[name]
                ))
            for name in sorted(self.internals_memory):
                assignments.append(tf.scatter_update(
                    ref=self.internals_memory[name],
                    indices=sorted_indices,
                    updates=self.internals_memory[name]
                ))
            assignments.append(tf.scatter_update(
                ref=self.reward_memory,
                indices=sorted_indices,
                updates=self.reward_memory
            ))

        # 6. Reset buffer index and increment memory index by inserted elements.
        with tf.control_dependencies(control_inputs=assignments):
            assignments = list()
            # Decrement pointer of last elements used.
            assignments.append(tf.assign_sub(ref=self.buffer_index, value=self.last_batch_buffer_elems))

            # Keep track of memory size as to know whether we can sample from the main memory.
            # Since the memory pointer can set to 0, we want to know if we are at capacity.
            total_inserted_elements = self.memory_size + self.last_batch_buffer_elems
            assignments.append(tf.assign(
                ref=self.memory_size,
                value=tf.minimum(x=total_inserted_elements, y=self.capacity))
            )

            # Update memory insertion index.
            assignments.append(tf.assign(ref=self.memory_index, value=memory_end_index))

            # Reset batch indices.
            assignments.append(tf.assign(
                ref=self.batch_indices,
                value=tf.zeros(shape=tf.shape(self.batch_indices), dtype=tf.int32)
            ))

        with tf.control_dependencies(control_inputs=assignments):
            return util.no_operation()

    # These are not supported for prioritized replay currently.
    def tf_retrieve_episodes(self, n):
        pass

    def tf_retrieve_sequences(self, n, sequence_length):
        pass
