import numpy as np
from tensorforce.exceptions.tensorforce_exceptions import ArgumentMustBePositiveException
from tensorforce.util.experiment_util import global_seed

"""
Replay memory to store observations and sample
mini batches for training from.
"""


class ReplayMemory(object):
    def __init__(self,
                 capacity,
                 state_shape,
                 state_type,
                 action_shape,
                 action_type,
                 reward_type,
                 concat=False,
                 concat_length=1,
                 deterministic_mode=False):
        """
        Initializes a replay memory.

        :param capacity: Memory size
        :param state_shape: Shape of state tensor
        :param state_type: Data type of state tensor
        :param action_shape: Shape of action tensor
        :param action_type: Data type of action tensor
        :param reward_type: Data type of reward function
        :param concat: Whether to apply preprocessing to satisfy Markov property -
        for some environments, single frames do not satisfy the Markov property but
        a concatenation of frames (for Atari 4) does
        :param concat_length: State preprocessor function sigma, here given as
        length to satisfy Markov property, default 1 means no concatenation of states.
        :param deterministic_mode: If true, global random number generation
        is controlled by passing the same seed to all generators, if false,
        no seed is used for sampling.
        """

        self.step_count = 0
        self.capacity = capacity
        self.size = 0
        self.concat = concat
        self.concat_length = concat_length

        # Explicitly set data types for every tensor to make for easier adjustments
        # if backend precision changes
        self.state_shape = state_shape
        self.state_type = state_type
        self.action_shape = action_shape
        self.action_type = action_type
        self.reward_type = reward_type

        self.states = np.zeros((capacity,) + state_shape, dtype=state_type)
        self.actions = np.zeros((capacity,) + action_shape, dtype=action_type)
        self.rewards = np.zeros(capacity, dtype=reward_type)
        self.terminals = np.zeros(capacity, dtype='bool')

        if deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        # Indices to control sampling
        self.bottom = 0
        self.top = 0

    def add_experience(self, state, action, reward, terminal):
        """
        Inserts an experience tuple to the memory.

        :param state: State observed.
        :param action: Action(s) taken
        :param reward: Reward seen after taking action
        :param terminal: Boolean whether episode ended
        :return:
        """

        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminals[self.top] = terminal

        if self.size == self.capacity:
            self.bottom = (self.bottom + 1) % self.capacity
        else:
            self.size += 1
        self.top = (self.top + 1) % self.capacity

    def sample_batch(self, batch_size):
        """
        Sample a mini batch of stored experiences
        :param batch_size:
        :return: A Tensor containing experience tuples of length batch_size

        """

        if batch_size < 0:
            raise ArgumentMustBePositiveException('Batch size must be positive')

        batch_states = np.zeros((batch_size, self.concat_length) + self.state_shape,
                                dtype=self.state_type)
        batch_actions = np.zeros((batch_size, self.action_shape), dtype=self.action_type)
        batch_rewards = np.zeros(batch_size, dtype=self.reward_type)
        batch_next_states = np.zeros((batch_size, self.sigma_length) + self.state_shape,
                                     dtype=self.state_type)
        batch_terminals = np.zeros(batch_size, dtype='bool')

        for i in xrange(batch_size):
            start_index = self.random.randint(self.bottom,
                                              self.bottom + self.size - self.concat_length)
            end_index = start_index
            if self.concat:
                state_index = np.arange(start_index, self.concat_length, 1)
                end_index = start_index + self.concat_length - 1
            else:
                state_index = start_index

            # Either range or single index depending on whether concatenation is active
            next_state_index = state_index + 1

            # Skip if concatenated index is between episodes
            if self.concat and np.any(self.terminals.take(state_index[0:-1], mode='wrap')):
                continue

            batch_states[i] = self.states.take(state_index, axis=0, mode='wrap')
            batch_actions[i] = self.actions.take(end_index, mode='wrap')
            batch_rewards[i] = self.rewards.take(end_index, mode='wrap')
            batch_next_states[i] = self.states.take(next_state_index, axis=0, mode='wrap')

        return dict(
            states=batch_states,
            actions=batch_actions,
            rewards=batch_rewards,
            next_states=batch_next_states,
            terminals=batch_terminals
        )
