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

from collections import OrderedDict

import tensorflow as tf

from tensorforce import TensorforceError, util
import tensorforce.core
from tensorforce.core import Module, parameter_modules
from tensorforce.core.parameters import Parameter


class Layer(Module):
    """
    Base class for neural network layers.

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    layers = None

    def __init__(self, name, input_spec=None, summary_labels=None, l2_regularization=None):
        super().__init__(
            name=name, summary_labels=summary_labels, l2_regularization=l2_regularization
        )

        self.input_spec = self.default_input_spec()
        self.input_spec = util.valid_value_spec(
            value_spec=self.input_spec, accept_underspecified=True, return_normalized=True
        )

        if input_spec is not None:
            input_spec = util.valid_value_spec(
                value_spec=input_spec, accept_underspecified=True, return_normalized=True
            )

            self.input_spec = util.unify_value_specs(
                value_spec1=self.input_spec, value_spec2=input_spec
            )

        # Copy so that spec can be modified
        self.output_spec = self.get_output_spec(input_spec=dict(self.input_spec))
        self.output_spec = util.valid_value_spec(
            value_spec=self.output_spec, accept_underspecified=True, return_normalized=True
        )

        # Register layer globally
        if Layer.layers is None:
            Layer.layers = OrderedDict()
        # if self.name in Layer.layers:
        #     raise TensorforceError.unexpected()
        Layer.layers[self.name] = self

    def default_input_spec(self):
        """
        Returns the general, context-independent input tensor specification of this layer.

        Returns:
            General input tensor specification.
        """
        raise NotImplementedError

    def get_output_spec(self, input_spec):
        """
        Returns the output tensor specification for a given input tensor specification.

        Args:
            input_spec (specification): Input tensor specification.

        Returns:
            Output tensor specification.
        """
        return input_spec

    def add_module(self, *args, **kwargs):
        layer = super().add_module(*args, **kwargs)

        if not isinstance(layer, (Layer, Parameter)):
            raise TensorforceError.type(name='layer', argument='sub-module', value=layer)

        return layer

    def tf_apply(self, x):
        return x

    def create_tf_function(self, name, tf_function):
        if tf_function.__name__ == 'tf_apply':

            def validated_tf_function(x):
                if self.input_spec is not None and \
                        not util.is_consistent_with_value_spec(value_spec=self.input_spec, x=x):
                    raise TensorforceError("Invalid input arguments for tf_apply.")

                x = tf_function(x=x)

                if self.output_spec is not None and \
                        not util.is_consistent_with_value_spec(value_spec=self.output_spec, x=x):
                    raise TensorforceError("Invalid output arguments for tf_apply.")

                return x

            return super().create_tf_function(name=name, tf_function=validated_tf_function)

        else:
            return super().create_tf_function(name=name, tf_function=tf_function)


class TransformationBase(Layer):
    """
    Base class for transformation layers.

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: false).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: none).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        is_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Additional arguments for potential parent class.
    """

    def __init__(
        self, name, size, bias=False, activation=None, dropout=0.0, is_trainable=True,
        input_spec=None, summary_labels=None, l2_regularization=None, **kwargs
    ):
        self.squeeze = (size == 0)
        self.size = max(size, 1)
        self.activation = None
        self.dropout = None

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization, **kwargs
        )

        self.bias = bias
        if activation is None:
            self.activation = None
        else:
            self.activation = self.add_module(
                name=(self.name + '-activation'), module='activation',
                modules=tensorforce.core.layer_modules, nonlinearity=activation,
                input_spec=self.output_spec
            )
        if dropout is None or dropout == 0.0:
            self.dropout = None
        else:
            self.dropout = self.add_module(
                name=(self.name + '-dropout'), module='dropout',
                modules=tensorforce.core.layer_modules, rate=dropout, input_spec=self.output_spec
            )
        self.is_trainable = is_trainable

    def specify_input_output_spec(self, input_spec):
        super().specify_input_output_spec(input_spec=input_spec)

        if self.activation is not None:
            self.activation.specify_input_output_spec(input_spec=self.output_spec)
            assert self.activation.output_spec == self.output_spec
        if self.dropout is not None:
            self.dropout.specify_input_output_spec(input_spec=self.output_spec)
            assert self.dropout.output_spec == self.output_spec

    def tf_initialize(self):
        super().tf_initialize()

        if self.bias:
            self.bias = self.add_variable(
                name='bias', dtype='float', shape=(self.size,), is_trainable=self.is_trainable,
                initializer=('zeros' if self.is_trainable else 'normal')
            )

        else:
            self.bias = None

    def tf_apply(self, x):
        # shape = self.get_output_spec()['shape']
        # if self.squeeze:
        #     shape = shape + (1,)
        # if util.dtype(x=x) != 'float' or util.shape(x=x)[1:] != shape:
        #     raise TensorforceError("Invalid input tensor for generic layer: {}.".format(x))

        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias)

        if self.squeeze:
            x = tf.squeeze(input=x, axis=-1)

        if self.activation is not None:
            x = self.activation.apply(x=x)

        if self.dropout is not None:
            x = self.dropout.apply(x=x)

        return x


class TemporalLayer(Layer):
    """
    Base class for temporal layers, i.e. layers with a temporal dependency on previous states.
    """

    def __init__(
        self, name, processing, dependency_horizon, input_spec=None, summary_labels=None,
        l2_regularization=None, **kwargs
    ):
        """
        Temporal layer constructor.

        Args:
            processing ('cumulative' | 'iterative'): Temporal processing type (**required**).
            dependency_horizon (parameter, long >= 0): (**required**).
            kwargs: Additional arguments for potential parent class.
        """
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization, **kwargs
        )

        if processing not in ('cumulative', 'iterative'):
            raise TensorforceError.unexpected()

        self.processing = processing

        self.dependency_horizon = self.add_module(
            name='dependency-horizon', module=dependency_horizon, modules=parameter_modules,
            is_trainable=False, dtype='long'
        )

    def tf_apply(self, x, initial=None):
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        dependency_starts = Module.retrieve_tensor(name='dependency_starts')
        dependency_lengths = Module.retrieve_tensor(name='dependency_lengths')
        if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
            batch_size = tf.shape(input=dependency_starts, out_type=util.tf_dtype(dtype='long'))[0]
        else:
            batch_size = tf.dtypes.cast(
                x=tf.shape(input=dependency_starts)[0], dtype=util.tf_dtype(dtype='long')
            )
        zeros = tf.zeros(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
        ones = tf.ones(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
        # maximum_iterations = tf.math.reduce_max(input_tensor=lengths, axis=0)
        horizon = self.dependency_horizon.value() + one  # including 0th step
        starts = dependency_starts + tf.maximum(x=(dependency_lengths - horizon), y=zeros)
        lengths = dependency_lengths - tf.maximum(x=(dependency_lengths - horizon), y=zeros)
        horizon = tf.minimum(x=horizon, y=tf.math.reduce_max(input_tensor=lengths, axis=0))

        if self.processing == 'cumulative':

            def body(indices, remaining, xs):
                current_x = tf.gather(params=x, indices=indices)
                current_x = tf.expand_dims(input=current_x, axis=1)
                xs = tf.concat(values=(xs, current_x), axis=1)
                remaining -= tf.where(condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones)
                indices += tf.where(condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones)
                return indices, remaining, xs

            initial_xs = tf.zeros(
                shape=((batch_size, 0) + self.output_spec['shape']),
                dtype=util.tf_dtype(dtype=self.output_spec['type'])
            )

            final_indices, final_remaining, final_xs = self.while_loop(
                cond=util.tf_always_true, body=body, loop_vars=(starts, lengths, initial_xs),
                back_prop=True, maximum_iterations=horizon
            )

            # initial_xs = tf.gather(params=x, indices=starts)
            # initial_xs = tf.expand_dims(input=initial_xs, axis=1)
            # missing = tf.expand_dims(input=horizon, axis=0) - lengths
            # missing -= tf.where(condition=tf.math.equal(x=missing, y=zeros), x=zeros, y=ones)
            # starts += tf.where(condition=tf.math.equal(x=missing, y=zeros), x=ones, y=zeros)

            # final_indices, final_counter, final_xs = self.while_loop(
            #     cond=util.tf_always_true, body=body, loop_vars=(starts, missing, initial_xs),
            #     back_prop=True, maximum_iterations=(horizon - one)
            # )

        elif self.processing == 'iterative':

            def body(indices, remaining, current_x, current_aggregates):
                current_x = tf.gather(params=x, indices=indices)
                next_x, next_aggregates = self.iterative_step(
                    x=current_x, previous=current_aggregates
                )
                with tf.control_dependencies(control_inputs=(current_x, next_x)):
                    is_finished = tf.math.equal(x=remaining, y=zeros)
                    if isinstance(next_aggregates, dict):
                        for name, current_aggregate, next_aggregate in util.zip_items(
                            current_aggregates, next_aggregates
                        ):
                            condition = is_finished
                            for _ in range(util.rank(x=current_aggregate) - 1):
                                condition = tf.expand_dims(input=condition, axis=1)
                            next_aggregates[name] = tf.where(
                                condition=condition, x=current_aggregate, y=next_aggregate
                            )
                    else:
                        condition = is_finished
                        for _ in range(util.rank(x=current_aggregates) - 1):
                            condition = tf.expand_dims(input=condition, axis=1)
                        next_aggregates = tf.where(
                            condition=condition, x=current_aggregates, y=next_aggregates
                        )
                    remaining -= tf.where(condition=is_finished, x=zeros, y=ones)
                    indices += tf.where(
                        condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones
                    )
                return indices, remaining, next_x, next_aggregates

            initial_x = tf.zeros(
                shape=((batch_size,) + self.output_spec['shape']),
                dtype=util.tf_dtype(dtype=self.output_spec['type'])
            )

            if initial is None:
                initial_aggregates = self.initial_values()
            else:
                initial_aggregates = initial

            final_indices, final_remaining, final_x, final_aggregates = self.while_loop(
                cond=util.tf_always_true, body=body,
                loop_vars=(starts, lengths, initial_x, initial_aggregates), back_prop=True,
                maximum_iterations=horizon
            )

        # assertions = [
        #     tf.debugging.assert_equal(
        #         x=final_indices, y=(tf.math.cumsum(x=dependency_lengths) - ones)
        #     ),
        #     tf.debugging.assert_equal(
        #         x=tf.math.reduce_sum(input_tensor=final_remaining, axis=0), y=zero
        #     )
        # ]

        # with tf.control_dependencies(control_inputs=assertions):
        if self.processing == 'cumulative':
            return super().tf_apply(x=self.cumulative_apply(xs=final_xs))
        elif self.processing == 'iterative':
            if initial is None:
                return util.identity_operation(x=super().tf_apply(x=final_x))
            else:
                return util.identity_operation(x=super().tf_apply(x=final_x)), final_aggregates

    def tf_cumulative_apply(self, xs):
        raise NotImplementedError

    def tf_initial_values(self):
        raise NotImplementedError

    def tf_iterative_step(self, x, previous):
        raise NotImplementedError

    def create_tf_function(self, name, tf_function):
        if tf_function.__name__ == 'tf_apply':

            def validated_tf_function(x, initial=None):
                if not util.is_consistent_with_value_spec(value_spec=self.input_spec, x=x):
                    raise TensorforceError("Invalid input arguments for tf_apply.")

                # initial spec!

                if initial is None:
                    x = tf_function(x=x)
                else:
                    x, final = tf_function(x=x, initial=initial)

                if not util.is_consistent_with_value_spec(value_spec=self.output_spec, x=x):
                    raise TensorforceError("Invalid output arguments for tf_apply.")

                if initial is None:
                    return x
                else:
                    return x, final

            return super().create_tf_function(name=name, tf_function=validated_tf_function)

        elif tf_function.__name__ == 'tf_cumulative_apply':

            def validated_tf_function(xs):
                x = xs[:, 0, :]
                if not util.is_consistent_with_value_spec(value_spec=self.input_spec, x=x):
                    raise TensorforceError("Invalid input arguments for tf_apply.")

                x = tf_function(xs=xs)

                if not util.is_consistent_with_value_spec(value_spec=self.output_spec, x=x):
                    raise TensorforceError("Invalid output arguments for tf_apply.")

                return x

            return super().create_tf_function(name=name, tf_function=validated_tf_function)

        elif tf_function.__name__ == 'tf_iterative_step':

            def validated_tf_function(x, previous):
                if not util.is_consistent_with_value_spec(value_spec=self.input_spec, x=x):
                    raise TensorforceError("Invalid input arguments for tf_apply.")

                # previous spec!

                x, previous = tf_function(x=x, previous=previous)

                if not util.is_consistent_with_value_spec(value_spec=self.output_spec, x=x):
                    raise TensorforceError("Invalid output arguments for tf_apply.")

                return x, previous

            return super().create_tf_function(name=name, tf_function=validated_tf_function)

        else:
            return super().create_tf_function(name=name, tf_function=tf_function)


class StatefulLayer(TemporalLayer):
    """
    Base class for stateful layers, i.e. layers with a temporally evolving internal state.
    """

    def __init__(
        self, name, optimization_horizon, input_spec=None, summary_labels=None,
        l2_regularization=None, **kwargs
    ):
        """
        Stateful layer constructor.

        Args:
            optimization_horizon (parameter, long > 0): (**required**).
            kwargs: Additional arguments for potential parent class.
        """
        super().__init__(
            name=name, processing='iterative', dependency_horizon=optimization_horizon,
            input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization, **kwargs
        )

    @classmethod
    def internals_spec(cls, layer=None, **kwargs):
        raise NotImplementedError

    def internals_init(self):
        raise NotImplementedError

    # internals spec below!

    # def create_tf_function(self, name, tf_function):
    #     # if name[-6:] != '.apply':
    #     if tf_function.__name__ == 'tf_apply' or tf_function.__name__ == 'tf_apply_step':

    #         def validated_tf_function(x, previous):
    #             if not util.is_consistent_with_value_spec(value_spec=self.input_spec, x=x):
    #                 raise TensorforceError("Invalid input arguments for tf_apply.")
    #             if not all(
    #                 util.is_consistent_with_value_spec(value_spec=spec, x=previous[name])
    #                 for name, spec in self.__class__.internals_spec(layer=self).items()
    #             ):
    #                 raise TensorforceError("Invalid input arguments for tf_apply.")

    #             x, previous = tf_function(x=x, previous=previous)

    #             if not util.is_consistent_with_value_spec(value_spec=self.output_spec, x=x):
    #                 raise TensorforceError("Invalid output arguments for tf_apply.")
    #             if not all(
    #                 util.is_consistent_with_value_spec(value_spec=spec, x=previous[name])
    #                 for name, spec in self.__class__.internals_spec(layer=self).items()
    #             ):
    #                 raise TensorforceError("Invalid input arguments for tf_apply.")

    #             return x, previous

    #         return super().create_tf_function(name=name, tf_function=validated_tf_function)

    #     else:
    #         return super().create_tf_function(name=name, tf_function=tf_function)


    # def tf_apply(self, x, **internals):

    #     # optimization = tf.math.logical_not(x=Module.retrieve_tensor(name='optimization'))

    #     # def true_fn():
    #     batch_size = tf.shape(
    #         input=next(iter(internals.values())), out_type=util.tf_dtype(dtype='long')
    #     )[0]
    #     zeros = tf.zeros(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
    #     ones = tf.ones(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))

    #     def body(indices, remaining, current_x, current_internals):
    #         current_x = tf.gather(params=x, indices=indices)
    #         next_x, next_internals = self.apply_step(x=current_x, **current_internals)
    #         is_finished = tf.math.equal(x=remaining, y=zeros)
    #         for name, internal, next_internal in util.zip_items(current_internals, next_internals):
    #             next_internals[name] = tf.where(
    #                 condition=is_finished, x=internal, y=next_internal
    #             )
    #         remaining -= tf.where(condition=is_finished, x=zeros, y=ones)
    #         indices += tf.where(condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones)
    #         return indices, remaining, next_x, next_internals

    #     starts = Module.retrieve_tensor(name='sequence_starts')
    #     lengths = Module.retrieve_tensor(name='sequence_lengths')
    #     initial_x = tf.gather(params=x, indices=starts)  # could be constant zeros!
    #     maximum_iterations = tf.math.reduce_max(input_tensor=lengths, axis=0)
    #     final_indices, final_remaining, final_x, final_internals = self.while_loop(
    #         cond=util.tf_always_true, body=body, loop_vars=(starts, lengths, initial_x, internals),
    #         back_prop=True, maximum_iterations=maximum_iterations
    #     )

    #     assertions = [
    #         tf.debugging.assert_equal(x=final_indices, y=(tf.math.cumsum(x=lengths) - ones)),
    #         tf.debugging.assert_equal(
    #             x=tf.math.reduce_sum(input_tensor=final_remaining, axis=0),
    #             y=tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
    #         )
    #     ]

    #     with tf.control_dependencies(control_inputs=assertions):
    #         return super().tf_apply(x=final_x), final_internals

    #     # return final_x, final_internals

    #     # def false_fn():
    #     #     return self.apply_step(x=x, **internals)

    #     # x, internals = self.cond(pred=optimization, true_fn=true_fn, false_fn=false_fn)

    #     # return super().tf_apply(x=x), internals
