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

import tensorflow as tf

from tensorforce import util
from tensorforce.core.networks import Layer


class Network(object):
    """
    Base class for neural networks.
    """

    def __init__(self, scope='network', summary_labels=None):
        """
        Neural network.
        """
        self.summary_labels = set(summary_labels or ())

        self.variables = dict()
        self.all_variables = dict()
        self.named_tensors = dict()

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

        self.apply = tf.make_template(
            name_=(scope + '/apply'),
            func_=self.tf_apply,
            custom_getter_=custom_getter
        )
        self.regularization_loss = tf.make_template(
            name_=(scope + '/regularization-loss'),
            func_=self.tf_regularization_loss,
            custom_getter_=custom_getter
        )

    def tf_apply(self, x, internals, update, return_internals=False):
        """
        Creates the TensorFlow operations for applying the network to the given input.

        Args:
            x: Network input tensor or dict of input tensors.
            internals: List of prior internal state tensors
            update: Boolean tensor indicating whether this call happens during an update.
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

    def internals_spec(self):
        """
        Returns the internal states specification.

        Returns:
            Internal states specification
        """
        return dict()

    def get_variables(self, include_nontrainable=False):
        """
        Returns the TensorFlow variables used by the network.

        Returns:
            List of variables
        """
        if include_nontrainable:
            return [self.all_variables[key] for key in sorted(self.all_variables)]
        else:
            return [self.variables[key] for key in sorted(self.variables)]

    def get_named_tensor(self, name):
        """
        Returns a named tensor if available.

        Returns:
            valid: True if named tensor found, False otherwise
            tensor: If valid, will be a tensor, otherwise None
        """
        if name in self.named_tensors:
            return True, self.named_tensors[name]
        else:
            return False, None

    def get_list_of_named_tensor(self):
        """
        Returns a list of the names of tensors available.

        Returns:
            List of the names of tensors available.
        """
        return list(self.named_tensors)

    def set_named_tensor(self, name, tensor):
        self.named_tensors[name] = tensor

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
        """
        Layer-based network.
        """
        super(LayerBasedNetwork, self).__init__(scope=scope, summary_labels=summary_labels)
        self.layers = list()

    def add_layer(self, layer):
        self.layers.append(layer)

    def tf_regularization_loss(self):
        regularization_loss = super(LayerBasedNetwork, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        for layer in self.layers:
            regularization_loss = layer.regularization_loss()
            if regularization_loss is not None:
                losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def internals_spec(self):
        internals_spec = dict()
        for layer in self.layers:
            spec = layer.internals_spec()
            for name in sorted(spec):
                internals_spec['{}_{}'.format(layer.scope, name)] = spec[name]
        return internals_spec

    def get_variables(self, include_nontrainable=False):
        network_variables = super(LayerBasedNetwork, self).get_variables(
            include_nontrainable=include_nontrainable
        )
        layer_variables = [
            variable for layer in self.layers
            for variable in layer.get_variables(include_nontrainable=include_nontrainable)
        ]

        return network_variables + layer_variables


class LayeredNetwork(LayerBasedNetwork):
    """
    Network consisting of a sequence of layers, which can be created from a specification dict.
    """

    def __init__(self, layers, scope='layered-network', summary_labels=()):
        """
        Single-stack layered network.

        Args:
            layers: List of layer specification dicts.
        """
        self.layers_spec = layers
        super(LayeredNetwork, self).__init__(scope=scope, summary_labels=summary_labels)

        self.parse_layer_spec(layer_spec=self.layers_spec, layer_counter=Counter())

    def parse_layer_spec(self, layer_spec, layer_counter):
        if isinstance(layer_spec, list):
            for layer_spec in layer_spec:
                self.parse_layer_spec(layer_spec=layer_spec, layer_counter=layer_counter)
        else:
            if isinstance(layer_spec['type'], str):
                name = layer_spec['type']
            else:
                name = 'layer'
            scope = name + str(layer_counter[name])
            layer_counter[name] += 1

            layer = Layer.from_spec(
                spec=layer_spec,
                kwargs=dict(named_tensors=self.named_tensors, scope=scope, summary_labels=self.summary_labels)
            )
            self.add_layer(layer=layer)

    def tf_apply(self, x, internals, update, return_internals=False):
        if isinstance(x, dict):
            self.named_tensors.update(x)
            if len(x) == 1:
                x = x[next(iter(sorted(x)))]

        next_internals = dict()
        for layer in self.layers:
            layer_internals = {name: internals['{}_{}'.format(layer.scope, name)] for name in layer.internals_spec()}

            if len(layer_internals) > 0:
                x, layer_internals = layer.apply(x=x, update=update, **layer_internals)
                for name in sorted(layer_internals):
                    next_internals['{}_{}'.format(layer.scope, name)] = layer_internals[name]

            else:
                x = layer.apply(x=x, update=update)

        if return_internals:
            return x, next_internals
        else:
            return x
