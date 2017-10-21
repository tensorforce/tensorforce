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
from __future__ import division
from __future__ import print_function

from collections import Counter
import json
import os

import tensorflow as tf

from tensorforce import util, TensorForceError
from tensorforce.core.networks import Layer


class Network(object):
    """
    Base class for neural networks.
    """

    def __init__(self, scope='network', summary_labels=None):
        self.summary_labels = set(summary_labels or ())

        self.variables = dict()
        self.all_variables = dict()
        self.summaries = list()

        with tf.name_scope(name=scope):
            def custom_getter(getter, name, registered=False, **kwargs):
                variable = getter(name=name, registered=True, **kwargs)
                if not registered:
                    self.all_variables[name] = variable
                    if kwargs.get('trainable', True):
                        self.variables[name] = variable
                    if 'variables' in self.summary_labels:
                        summary = tf.summary.histogram(name=name, values=variable)
                        self.summaries.append(summary)
                return variable

            self.apply = tf.make_template(
                name_='apply',
                func_=self.tf_apply,
                custom_getter_=custom_getter
            )
            self.regularization_loss = tf.make_template(
                name_='regularization-loss',
                func_=self.tf_regularization_loss,
                custom_getter_=custom_getter
            )

    def tf_apply(self, x, internals=(), return_internals=False):
        """
        Creates the TensorFlow operations for applying the network to the given input.

        Args:
            x: Network input tensor or dict of input tensors
            internals: List of prior internal state tensors
            return_internals: If true, also returns posterior internal state tensors

        Returns:
            Network output tensor, plus optionally list of posterior internal state tensors
        """
        raise NotImplementedError

    def tf_regularization_loss(self):
        """
        Creates the TensorFlow operations for the network regularization loss.

        Returns:
            Regularization loss tensor
        """
        return None

    def internal_inputs(self):
        """
        Returns the TensorFlow placeholders for internal state inputs.

        Returns:
            List of internal state input placeholders
        """
        return list()

    def internal_inits(self):
        """
        Returns the TensorFlow tensors for internal state initializations.

        Returns:
            List of internal state initialization tensors
        """
        return list()

    def get_variables(self, include_non_trainable=False):
        """
        Returns the TensorFlow variables used by the network.

        Returns:
            List of variables
        """
        if include_non_trainable:
            return [self.all_variables[key] for key in sorted(self.all_variables)]
        else:
            return [self.variables[key] for key in sorted(self.variables)]

    def get_summaries(self):
        """
        Returns the TensorFlow summaries reported by the network.

        Returns:
            List of summaries
        """
        return self.summaries

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a network from a specification dict.
        """
        network = util.get_object(
            obj=spec,
            default_object=LayeredNetwork,
            kwargs=kwargs
        )
        assert isinstance(network, Network)
        return network


class LayerBasedNetwork(Network):
    """
    Base class for networks using TensorForce layers.
    """

    def __init__(self, scope='layerbased-network', summary_labels=()):
        super(LayerBasedNetwork, self).__init__(scope=scope, summary_labels=summary_labels)
        self.layers = list()

    def add_layer(self, layer):
        self.layers.append(layer)

    def tf_regularization_loss(self):
        if super(LayerBasedNetwork, self).tf_regularization_loss() is None:
            losses = list()
        else:
            losses = [super(LayerBasedNetwork, self).tf_regularization_loss()]

        for layer in self.layers:
            if layer.regularization_loss() is not None:
                losses.append(layer.regularization_loss())

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def internal_inputs(self):
        internal_inputs = super(LayerBasedNetwork, self).internal_inputs()
        for layer in self.layers:
            internal_inputs.extend(layer.internal_inputs())
        return internal_inputs

    def internal_inits(self):
        internal_inits = super(LayerBasedNetwork, self).internal_inits()
        for layer in self.layers:
            internal_inits.extend(layer.internal_inits())
        return internal_inits

    def get_variables(self, include_non_trainable=False):
        network_variables = super(LayerBasedNetwork, self).get_variables(
            include_non_trainable=include_non_trainable
        )

        layer_variables = [
            variable for layer in self.layers
            for variable in layer.get_variables(include_non_trainable=include_non_trainable)
        ]

        return network_variables + layer_variables

    def get_summaries(self):
        return super(LayerBasedNetwork, self).get_summaries() + \
            [summary for layer in self.layers for summary in layer.get_summaries()]


class LayeredNetwork(LayerBasedNetwork):
    """
    Network consisting of a sequence of layers, which can be created from a specification dict.
    """

    def __init__(self, layers_spec, scope='layered-network', summary_labels=()):
        """
        Layered network.

        Args:
            layers_spec: List of layer specification dicts
        """
        super(LayeredNetwork, self).__init__(scope=scope, summary_labels=summary_labels)
        self.layers_spec = layers_spec
        layer_counter = Counter()

        with tf.name_scope(name=scope):
            for layer_spec in self.layers_spec:
                layer = Layer.from_spec(
                    spec=layer_spec,
                    kwargs=dict(scope=scope, summary_labels=summary_labels)
                )

                name = layer_spec['type'].__class__.__name__
                scope = name + str(layer_counter[name])
                layer_counter[name] += 1

                self.add_layer(layer=layer)

    def tf_apply(self, x, internals=(), return_internals=False):
        if isinstance(x, dict):
            if len(x) != 1:
                raise TensorForceError('Layered network must have only one input, but {} given.'.format(len(x)))
            x = next(iter(x.values()))

        internal_outputs = list()
        index = 0
        for layer in self.layers:
            layer_internals = [internals[index + n] for n in range(layer.num_internals)]
            index += layer.num_internals
            x = layer.apply(x, *layer_internals)

            if not isinstance(x, tf.Tensor):
                internal_outputs.extend(x[1])
                x = x[0]

        if return_internals:
            return x, internal_outputs
        else:
            return x

    @staticmethod
    def from_json(filename):
        """
        Creates a layer_networkd_builder from a JSON.

        Args:
            filename: Path to configuration

        Returns: A layered_network_builder function with layers generated from the JSON
        """
        path = os.path.join(os.getcwd(), filename)
        with open(path, 'r') as fp:
            config = json.load(fp=fp)
        return LayeredNetwork(layers_spec=config)
