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
import tensorforce.core.baselines


class Baseline(object):
    """
    Base class for baseline value functions.
    """

    def __init__(self, scope='baseline', summary_labels=None):
        """
        Baseline.
        """
        self.summary_labels = set(summary_labels or ())

        self.variables = dict()
        self.all_variables = dict()

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if registered:
                pass
            elif name in self.all_variables:
                assert variable is self.all_variables[name]
                if kwargs.get('trainable', True):
                    assert variable is self.variables[name]
                    if 'variables' in self.summary_labels:
                        tf.contrib.summary.histogram(name=name, tensor=variable)
            else:
                self.all_variables[name] = variable
                if kwargs.get('trainable', True):
                    self.variables[name] = variable
                    if 'variables' in self.summary_labels:
                        tf.contrib.summary.histogram(name=name, tensor=variable)
            return variable

        self.predict = tf.make_template(
            name_=(scope + '/predict'),
            func_=self.tf_predict,
            custom_getter_=custom_getter
        )
        self.reference = tf.make_template(
            name_=(scope + '/reference'),
            func_=self.tf_reference,
            custom_getter_=custom_getter
        )
        self.loss = tf.make_template(
            name_=(scope + '/loss'),
            func_=self.tf_loss,
            custom_getter_=custom_getter
        )
        self.regularization_loss = tf.make_template(
            name_=(scope + '/regularization-loss'),
            func_=self.tf_regularization_loss,
            custom_getter_=custom_getter
        )

    def tf_predict(self, states, internals, update):
        """
        Creates the TensorFlow operations for predicting the value function of given states.
        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            update: Boolean tensor indicating whether this call happens during an update.
        Returns:
            State value tensor
        """
        raise NotImplementedError

    def tf_reference(self, states, internals, reward, update):
        """
        Creates the TensorFlow operations for obtaining the reference tensor(s), in case of a
        comparative loss.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            reward: Reward tensor.
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            Reference tensor(s).
        """
        return None

    def tf_loss(self, states, internals, reward, update, reference=None):
        """
        Creates the TensorFlow operations for calculating the L2 loss between predicted
        state values and actual rewards.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            reward: Reward tensor.
            update: Boolean tensor indicating whether this call happens during an update.
            reference: Optional reference tensor(s), in case of a comparative loss.

        Returns:
            Loss tensor
        """
        prediction = self.predict(states=states, internals=internals, update=update)
        return tf.nn.l2_loss(t=(prediction - reward))

    def tf_regularization_loss(self):
        """
        Creates the TensorFlow operations for the baseline regularization loss/

        Returns:
            Regularization loss tensor
        """
        return None

    def get_variables(self, include_nontrainable=False):
        """
        Returns the TensorFlow variables used by the baseline.

        Returns:
            List of variables
        """
        if include_nontrainable:
            return [self.all_variables[key] for key in sorted(self.all_variables)]
        else:
            return [self.variables[key] for key in sorted(self.variables)]

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a baseline from a specification dict.
        """
        baseline = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.baselines.baselines,
            kwargs=kwargs
        )
        assert isinstance(baseline, Baseline)
        return baseline
