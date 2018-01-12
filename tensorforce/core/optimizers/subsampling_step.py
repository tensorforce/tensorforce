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
from tensorforce.core.optimizers import MetaOptimizer


class SubsamplingStep(MetaOptimizer):
    """
    The subsampling-step meta optimizer randomly samples a subset of batch instances to calculate  
    the optimization step of another optimizer.
    """

    def __init__(self, optimizer, fraction=0.1, summaries=None, summary_labels=None):
        """
        Creates a new subsampling-step meta optimizer instance.

        Args:
            optimizer: The optimizer which is modified by this meta optimizer.
            fraction: The fraction of instances of the batch to subsample.
        """
        super(SubsamplingStep, self).__init__(optimizer=optimizer, summaries=summaries, summary_labels=summary_labels)

        assert isinstance(fraction, float) and fraction > 0.0
        self.fraction = fraction

    def tf_step(
        self,
        time,
        variables,
        states,
        internals,
        actions,
        terminal,
        reward,
        next_states,
        next_internals,
        **kwargs
    ):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            states: Dictionary of batch state tensors.
            internals: List of batch internal tensors.
            actions: Dictionary of batch action tensors.
            terminal: Batch terminal tensor.
            reward: Batch reward tensor.
            next_states: Dictionary of batch successor state tensors.
            next_internals: List of batch posterior internal state tensors.
            **kwargs: Additional arguments passed on to the internal optimizer.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        batch_size = tf.shape(input=terminal)[0]
        num_samples = tf.cast(
            x=(self.fraction * tf.cast(x=batch_size, dtype=util.tf_dtype('float'))),
            dtype=util.tf_dtype('int')
        )
        num_samples = tf.maximum(x=num_samples, y=1)
        indices = tf.random_uniform(shape=(num_samples,), maxval=batch_size, dtype=tf.int32)

        subsampled_states = dict()
        for name, state in states.items():
            subsampled_states[name] = tf.gather(params=state, indices=indices)
        subsampled_internals = list()
        for n, internal in enumerate(internals):
            subsampled_internals.append(tf.gather(params=internal, indices=indices))
        subsampled_actions = dict()
        for name, action in actions.items():
            subsampled_actions[name] = tf.gather(params=action, indices=indices)
        subsampled_terminal = tf.gather(params=terminal, indices=indices)
        subsampled_reward = tf.gather(params=reward, indices=indices)
        if next_states is None:
            subsampled_next_states = None
            subsampled_next_internals = None
        else:
            subsampled_next_states = dict()
            for name, state in next_states.items():
                subsampled_next_states[name] = tf.gather(params=state, indices=indices)
            subsampled_next_internals = list()
            for n, internal in enumerate(next_internals):
                subsampled_next_internals.append(tf.gather(params=internal, indices=indices))

        return self.optimizer.step(
            time=time,
            variables=variables,
            states=subsampled_states,
            internals=subsampled_internals,
            actions=subsampled_actions,
            terminal=subsampled_terminal,
            reward=subsampled_reward,
            next_states=subsampled_next_states,
            next_internals=subsampled_next_internals,
            **kwargs
        )
