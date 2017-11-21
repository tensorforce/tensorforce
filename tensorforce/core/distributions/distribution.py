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
import tensorforce.core.distributions


class Distribution(object):
    """
    Base class for policy distributions.
    """

    def __init__(self, scope='distribution', summary_labels=None):
        self.summary_labels = set(summary_labels or ())

        self.variables = dict()
        self.all_variables = dict()
        self.summaries = list()

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if not registered:
                self.all_variables[name] = variable
                if kwargs.get('trainable', True) and not name.startswith('optimization'):
                    self.variables[name] = variable
                    if 'variables' in self.summary_labels:
                        summary = tf.summary.histogram(name=name, values=variable)
                        self.summaries.append(summary)
            return variable

        self.parameterize = tf.make_template(
            name_=(scope + '/parameterize'),
            func_=self.tf_parameterize,
            custom_getter_=custom_getter
        )
        self.sample = tf.make_template(
            name_=(scope + '/sample'),
            func_=self.tf_sample,
            custom_getter_=custom_getter
        )
        self.log_probability = tf.make_template(
            name_=(scope + '/log-probability'),
            func_=self.tf_log_probability,
            custom_getter_=custom_getter
        )
        self.entropy = tf.make_template(
            name_=(scope + '/entropy'),
            func_=self.tf_entropy,
            custom_getter_=custom_getter
        )
        self.kl_divergence = tf.make_template(
            name_=(scope + '/kl-divergence'),
            func_=self.tf_kl_divergence,
            custom_getter_=custom_getter
        )
        self.regularization_loss = tf.make_template(
            name_=(scope + '/regularization-loss'),
            func_=self.tf_regularization_loss,
            custom_getter_=custom_getter
        )

    def tf_parameterize(self, x):
        """
        Creates the TensorFlow operations for parameterizing a distribution conditioned on the  
        given input.

        Args:
            x: Input tensor which the distribution is conditioned on.

        Returns:
            Tuple of distribution parameter tensors.
        """
        raise NotImplementedError

    def tf_sample(self, distr_params, deterministic):
        """
        Creates the TensorFlow operations for sampling an action based on a distribution.

        Args:
            distr_params: Tuple of distribution parameter tensors.
            deterministic: Boolean input tensor indicating whether the maximum likelihood action  
            should be returned.

        Returns:
            Sampled action tensor.
        """
        raise NotImplementedError

    def tf_log_probability(self, distr_params, action):
        """
        Creates the TensorFlow operations for calculating the log probability of an action for a  
        distribution.

        Args:
            distr_params: Tuple of distribution parameter tensors.
            action: Action tensor.

        Returns:
            KL divergence tensor.
        """
        raise NotImplementedError

    def tf_entropy(self, distr_params):
        """
        Creates the TensorFlow operations for calculating the entropy of a distribution.

        Args:
            distr_params: Tuple of distribution parameter tensors.

        Returns:
            Entropy tensor.
        """
        raise NotImplementedError

    def tf_kl_divergence(self, distr_params1, distr_params2):
        """
        Creates the TensorFlow operations for calculating the KL divergence between two  
        distributions.

        Args:
            distr_params1: Tuple of parameter tensors for first distribution.
            distr_params2: Tuple of parameter tensors for second distribution.

        Returns:
            KL divergence tensor.
        """
        raise NotImplementedError

    def tf_regularization_loss(self):
        """
        Creates the TensorFlow operations for the distribution regularization loss.

        Returns:
            Regularization loss tensor.
        """
        return None

    def get_variables(self, include_non_trainable=False):
        """
        Returns the TensorFlow variables used by the distribution.

        Returns:
            List of variables.
        """
        if include_non_trainable:
            return [self.all_variables[key] for key in sorted(self.all_variables)]
        else:
            return [self.variables[key] for key in sorted(self.variables)]

    def get_summaries(self):
        """
        Returns the TensorFlow summaries reported by the distribution.

        Returns:
            List of summaries.
        """
        return self.summaries

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a distribution from a specification dict.
        """
        distribution = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.distributions.distributions,
            kwargs=kwargs
        )
        assert isinstance(distribution, Distribution)
        return distribution
