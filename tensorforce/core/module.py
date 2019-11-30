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
import importlib
import json
from math import sqrt
import os
import time

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util


class Module(object):
    """
    Base class for modules.

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    global_scope = None
    scope_stack = None
    while_counter = None
    cond_counter = None

    global_tensors_spec = None
    global_tensors = None  # per agent, main module, or so
    queryable_tensors = None

    global_summary_step = None

    @staticmethod
    def register_tensor(name, spec, batched):
        if '/' in name:
            raise TensorforceError.value(name='name', value=name)

        if Module.global_scope is None:  # ???
            raise TensorforceError.unexpected()

        scoped_name = name

        # if scoped_name in Module.global_tensors_spec:
        #     raise TensorforceError("Global tensor already exists: {}.".format(scoped_name))

        # optional? better to put in spec?
        spec = dict(spec)
        spec['batched'] = batched

        if scoped_name in Module.global_tensors_spec and \
                spec != Module.global_tensors_spec[scoped_name]:
            raise TensorforceError.mismatch(
                name='tensor-spec', value1=spec, value2=Module.global_tensors_spec[scoped_name]
            )

        if not util.valid_value_spec(value_spec=spec):
            raise TensorforceError.unexpected()

        if 'batched' in spec and spec['batched'] != batched:
            raise TensorforceError.unexpected()

        Module.global_tensors_spec[scoped_name] = spec

    @staticmethod
    def get_tensor_spec(name):
        if name not in Module.global_tensors_spec:
            raise TensorforceError.value(name='name', value=name)

        spec = dict(Module.global_tensors_spec[name])
        spec.pop('batched')

        return spec

    @staticmethod
    def update_tensor(name, tensor):
        if name not in Module.global_tensors_spec:
            raise TensorforceError("Global tensor is not registered: {}.".format(name))

        scoped_name = name
        spec = Module.global_tensors_spec[scoped_name]

        if not util.is_consistent_with_value_spec(value_spec=spec, x=tensor):
            raise TensorforceError("Invalid overwriting tensor: {}.".format(tensor))

        scoped_name = util.join_scopes(*Module.global_scope, name)

        previous = Module.global_tensors.get(scoped_name)
        Module.global_tensors[scoped_name] = tensor
        if Module.cond_counter == 0 and Module.while_counter == 0:
            Module.queryable_tensors[scoped_name] = tensor

        return previous

    @staticmethod
    def update_tensors(**kwargs):
        for name, tensor in kwargs.items():
            Module.update_tensor(name=name, tensor=tensor)

    @staticmethod
    def retrieve_tensor(name):
        if name not in Module.global_tensors_spec:
            raise TensorforceError("Global tensor is not registered: {}.".format(name))

        for n in range(len(Module.global_scope) + 1):
            partial_scope = Module.global_scope[:len(Module.global_scope) - n]
            scoped_name = util.join_scopes(*partial_scope, name)
            if scoped_name in Module.global_tensors:
                break
        else:
            raise TensorforceError("Global tensor is not set: {}.".format(name))

        return Module.global_tensors[scoped_name]

    is_add_module = False

    # Set internal attributes
    set_parent = None

    # Inherit arguments
    inherit_l2_regularization = None
    inherit_summary_labels = None

    def __init__(self, name, device=None, summary_labels=None, l2_regularization=None):
        # Internal attributes
        self.parent = Module.set_parent
        # self.scope = None
        self.is_subscope = None
        self.modules = OrderedDict()
        self.trainable_modules = OrderedDict()
        self.saved_modules = OrderedDict()
        self.is_initialized = False
        self.variables = None
        self.trainable_variables = None
        self.saved_variables = None
        self.output_tensors = None
        self.query_tensors = None
        self.available_summaries = None

        # name
        if not util.is_valid_name(name=name):
            raise TensorforceError.value(name='module', argument='name', value=name)
        # summary_labels
        if summary_labels is not None and \
                not all(isinstance(label, str) for label in summary_labels):
            raise TensorforceError.type(
                name='module', argument='summary_labels', value=summary_labels
            )
        # device
        # ???

        self.name = name
        self.device = device
        if summary_labels is None:
            # Otherwise inherit arguments
            self.summary_labels = Module.inherit_summary_labels
        elif summary_labels == 'all':
            self.summary_labels = summary_labels
        else:
            self.summary_labels = set(summary_labels)

        if not Module.is_add_module:
            Module.global_scope = list()
            Module.global_tensors_spec = OrderedDict()

        if Module.inherit_l2_regularization is None and l2_regularization is None:
            self.l2_regularization = None  # otherwise infinite recursion
        elif l2_regularization is not None:
            from tensorforce.core import parameter_modules
            self.l2_regularization = None  # for first module
            self.l2_regularization = self.add_module(
                name='l2-regularization', module=l2_regularization, modules=parameter_modules,
                is_trainable=False, dtype='float'
            )
        else:
            # Otherwise inherit arguments
            self.l2_regularization = Module.inherit_l2_regularization

    def tf_initialize(self):
        pass

    def tf_regularize(self):
        zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))

        if len(self.trainable_variables) == 0:
            regularization_loss = zero

        else:
            l2_regularization = self.l2_regularization.value()

            def no_l2_regularization():
                return zero

            def apply_l2_regularization():
                l2_variables = list()
                for variable in self.trainable_variables.values():
                    if variable.dtype != util.tf_dtype(dtype='float'):
                        variable = tf.dtypes.cast(x=variable, dtype=util.tf_dtype(dtype='float'))
                    l2_variables.append(tf.reduce_sum(input_tensor=tf.square(x=variable)))
                return l2_regularization * tf.math.add_n(inputs=l2_variables)

            skip_l2_regularization = tf.math.equal(x=l2_regularization, y=zero)
            regularization_loss = self.cond(
                pred=skip_l2_regularization, true_fn=no_l2_regularization,
                false_fn=apply_l2_regularization
            )

        for module in self.trainable_modules.values():
            regularization_loss += module.regularize()

        return regularization_loss

    def initialize(self):
        # Check whether module is already initialized
        if self.is_initialized:
            raise TensorforceError(message="Module already initialized.")

        # Set internal attributes
        self.is_initialized = True
        self.variables = OrderedDict()
        self.trainable_variables = OrderedDict()
        self.saved_variables = OrderedDict()
        self.output_tensors = dict()
        self.query_tensors = dict()
        self.available_summaries = set()

        if self.parent is None:
            Module.global_scope = list()
            Module.while_counter = 0
            Module.cond_counter = 0

            # Global timestep
            self.global_timestep = self.add_variable(
                name='global-timestep', dtype='long', shape=(), is_trainable=False,
                initializer='zeros', shared='global-timestep'
            )
            collection = self.graph.get_collection(name='global_step')
            if len(collection) == 0:
                self.graph.add_to_collection(name='global_step', value=self.global_timestep)

            if self.summarizer_spec is not None:
                with tf.name_scope(name='summarizer'):

                    directory = self.summarizer_spec['directory']
                    if os.path.isdir(directory):
                        directories = sorted(
                            d for d in os.listdir(directory)
                            if os.path.isdir(os.path.join(directory, d))
                            and d.startswith('summary-')
                        )
                    else:
                        os.makedirs(directory)
                        directories = list()
                    max_summaries = self.summarizer_spec.get('max-summaries', 5)
                    if len(directories) > max_summaries - 1:
                        for subdir in directories[:len(directories) - max_summaries + 1]:
                            subdir = os.path.join(directory, subdir)
                            os.remove(os.path.join(subdir, os.listdir(subdir)[0]))
                            os.rmdir(subdir)

                    logdir = os.path.join(directory, time.strftime('summary-%Y%m%d-%H%M%S'))
                    flush_millis = (self.summarizer_spec.get('flush', 10) * 1000)
                    self.summarizer = tf.summary.create_file_writer(
                        logdir=logdir, max_queue=None, flush_millis=flush_millis,
                        filename_suffix=None
                    )
                    self.summarizer_init = self.summarizer.init()
                    self.summarizer_flush = self.summarizer.flush()
                    self.summarizer_close = self.summarizer.close()

                    default_summarizer = self.summarizer.as_default()
                    default_summarizer.__enter__()

                    if self.summary_labels == 'all' or 'graph' in self.summary_labels:
                        pass

        # TensorFlow device and scope
        Module.global_scope.append(self.name)
        if self.device is not None:
            self.device = tf.device(device_name_or_function=self.device)
            self.device.__enter__()
        self.scope = tf.name_scope(name=self.name)

        with self.scope:
            if self.parent is None:
                # with tf.device(device_name_or_function=(self.global_model.device if self.global_model else self.device)):

                if self.summarizer_spec is not None:
                    Module.global_summary_step = 'timestep'
                    condition = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
                    record_summaries = tf.summary.record_if(condition=condition)
                    record_summaries.__enter__()

                # Global episode
                self.global_episode = self.add_variable(
                    name='global-episode', dtype='long', shape=(), is_trainable=False,
                    initializer='zeros', shared='global-episode'
                )

                # Global update
                self.global_update = self.add_variable(
                    name='global-update', dtype='long', shape=(), is_trainable=False,
                    initializer='zeros', shared='global-update'
                )

                Module.global_tensors = OrderedDict(
                    timestep=self.global_timestep, episode=self.global_episode,
                    update=self.global_update
                )

                if self.summarizer_spec is not None:
                    record_summaries.__exit__(None, None, None)

                    Module.global_summary_step = 'update'
                    if 'frequency' not in self.summarizer_spec or \
                            isinstance(self.summarizer_spec['frequency'], int):
                        condition = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))

                    elif 'variables' in self.summarizer_spec['frequency']:
                        step = Module.retrieve_tensor(name=Module.global_summary_step)
                        frequency = tf.constant(
                            value=self.summarizer_spec['frequency']['variables'],
                            dtype=util.tf_dtype(dtype='long')
                        )
                        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
                        condition = (
                            lambda: tf.math.equal(x=tf.math.mod(x=step, y=frequency), y=zero)
                        )

                    else:
                        condition = tf.constant(value=False, dtype=util.tf_dtype(dtype='bool'))

                    record_summaries = tf.summary.record_if(condition=condition)
                    record_summaries.__enter__()

            for module in self.modules.values():
                module.initialize()
            self.tf_initialize()

            if self.parent is None and self.summarizer_spec is not None:
                record_summaries.__exit__(None, None, None)
                Module.global_summary_step = None

        self.scope = None
        if self.device is not None:
            self.device.__exit__(None, None, None)
        Module.global_scope.pop()

        if self.parent is None:
            assert len(Module.global_scope) == 0
            Module.global_scope = None
            Module.while_counter = None
            Module.cond_counter = None
            Module.global_tensors = None

        # Internal TensorFlow functions, prefixed by 'tf_'
        for attribute in sorted(dir(self)):
            if attribute.startswith('tf_') and attribute != 'tf_initialize':
                function_name = attribute[3:]

                if not util.is_valid_name(name=function_name):
                    raise TensorforceError.value(name='TF-function name', value=function_name)
                if hasattr(self, function_name):
                    raise TensorforceError.exists(name='TF-function', value=function_name)

                tf_function = getattr(self, attribute)
                if not callable(tf_function):
                    raise TensorforceError.exists(name='TF-function', value=tf_function)

                function = self.create_tf_function(
                    name='{}.{}'.format(self.name, function_name), tf_function=tf_function
                )

                setattr(self, function_name, function)

        #  API TensorFlow functions, prefixed by 'api_'
        for attribute in sorted(dir(self)):
            if attribute.startswith('api_'):
                function_name = attribute[4:]
                assert hasattr(self, 'config')
                if self.config is not None and 'api_functions' in self.config and \
                        function_name not in self.config['api_functions']:
                    continue

                if function_name == 'act':
                    Module.global_summary_step = 'timestep'
                elif function_name in ('observe', 'experience'):
                    Module.global_summary_step = 'episode'
                elif function_name == 'update':
                    Module.global_summary_step = 'update'

                if self.summarizer_spec is not None:
                    if 'frequency' not in self.summarizer_spec:
                        condition = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))

                    elif isinstance(self.summarizer_spec['frequency'], int):
                        if function_name == 'act':
                            step = self.global_timestep
                            frequency = tf.constant(
                                value=self.summarizer_spec['frequency'],
                                dtype=util.tf_dtype(dtype='long')
                            )
                            zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
                            condition = (
                                lambda: tf.math.equal(x=tf.math.mod(x=step, y=frequency), y=zero)
                            )
                        elif function_name == 'reset':
                            condition = tf.constant(value=False, dtype=util.tf_dtype(dtype='bool'))
                        else:
                            condition = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))

                    elif function_name in self.summarizer_spec['frequency']:
                        if function_name == 'act':
                            step = self.global_timestep
                        elif function_name in ('observe', 'experience'):
                            step = self.global_episode
                        elif function_name == 'update':
                            step = self.global_update
                        elif function_name == 'reset':
                            raise TensorforceError.unexpected()
                        else:
                            raise TensorforceError.unexpected()
                        frequency = tf.constant(
                            value=self.summarizer_spec['frequency'][function_name],
                            dtype=util.tf_dtype(dtype='long')
                        )
                        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
                        condition = (
                            lambda: tf.math.equal(x=tf.math.mod(x=step, y=frequency), y=zero)
                        )

                    else:
                        condition = tf.constant(value=False, dtype=util.tf_dtype(dtype='bool'))

                    record_summaries = tf.summary.record_if(condition=condition)
                    record_summaries.__enter__()

                if not util.is_valid_name(name=function_name):
                    raise TensorforceError.value(name='API-function name', value=function_name)
                if hasattr(self, function_name):
                    raise TensorforceError.exists(name='API-function', value=function_name)

                api_function = getattr(self, attribute)
                if not callable(api_function):
                    raise TensorforceError.exists(name='API-function', value=tf_function)

                function = self.create_api_function(
                    name='{}.{}'.format(self.name, function_name), api_function=api_function
                )

                setattr(self, function_name, function)

                if self.summarizer_spec is not None:
                    record_summaries.__exit__(None, None, None)
                    Module.global_summary_step = None

        if self.parent is None:
            self.graph_summary = None  # TODO!
            if self.summarizer_spec is not None:
                default_summarizer.__exit__(None, None, None)

    def create_tf_function(self, name, tf_function):
        # Call internal TensorFlow function
        def fn(*args, **kwargs):
            if self.is_subscope:
                Module.global_scope.append(self.name)
            if self.device is not None:
                self.device.__enter__()
            scope = tf.name_scope(name=name)
            Module.scope_stack.append(scope)
            scope.__enter__()
            results = tf_function(*args, **kwargs)
            scope.__exit__(None, None, None)
            Module.scope_stack.pop()
            if self.device is not None:
                self.device.__exit__(None, None, None)
            if self.is_subscope:
                Module.global_scope.pop()
            return results

        return fn

    def create_api_function(self, name, api_function):
        # Call API TensorFlow function
        Module.global_scope = list()
        Module.scope_stack = list()
        Module.while_counter = 0
        Module.cond_counter = 0
        Module.global_tensors = OrderedDict()
        Module.queryable_tensors = OrderedDict()

        if self.device is not None:
            self.device.__enter__()
        scope = tf.name_scope(name=name)
        Module.scope_stack.append(scope)
        scope.__enter__()

        results = api_function()
        self.output_tensors[name[name.index('.') + 1:]] = sorted(
            x.name[len(name) + 1: -9] for x in util.flatten(xs=results)
        )

        # Function-level identity operation for retrieval
        query_tensors = set()
        for scoped_name, tensor in Module.queryable_tensors.items():
            util.identity_operation(x=tensor, operation_name=(scoped_name + '-output'))
            assert scoped_name not in query_tensors
            query_tensors.add(scoped_name)
        self.query_tensors[name[name.index('.') + 1:]] = sorted(query_tensors)

        scope.__exit__(None, None, None)
        Module.scope_stack.pop()
        if self.device is not None:
            self.device.__exit__(None, None, None)

        assert len(Module.global_scope) == 0
        Module.global_scope = None
        assert len(Module.scope_stack) == 0
        Module.scope_stack = None
        Module.while_counter = None
        Module.cond_counter = None
        Module.global_tensors = None
        Module.queryable_tensors = None

        def fn(query=None, **kwargs):
            # Feed_dict dictionary
            feed_dict = dict()
            for key, arg in kwargs.items():
                if arg is None:
                    continue
                elif isinstance(arg, dict):
                    # Support single nesting (for states, internals, actions)
                    for key, arg in arg.items():
                        feed_dict[util.join_scopes(self.name, key) + '-input:0'] = arg
                else:
                    feed_dict[util.join_scopes(self.name, key) + '-input:0'] = arg
            if not all(isinstance(x, str) and x.endswith('-input:0') for x in feed_dict):
                raise TensorforceError.unexpected()

            # Fetches value/tuple
            fetches = util.fmap(function=(lambda x: x.name), xs=results)
            if query is not None:
                # If additional tensors are to be fetched
                query = util.fmap(
                    function=(lambda x: util.join_scopes(name, x) + '-output:0'), xs=query
                )
                if util.is_iterable(x=fetches):
                    fetches = tuple(fetches) + (query,)
                else:
                    fetches = (fetches, query)
            if not util.reduce_all(
                predicate=(lambda x: isinstance(x, str) and x.endswith('-output:0')), xs=fetches
            ):
                raise TensorforceError.unexpected()

            # TensorFlow session call
            fetched = self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)

            return fetched

        return fn

    def cond(self, pred, true_fn, false_fn):

        def true_fn_wrapper():
            for scope in Module.scope_stack:
                scope.__enter__()
            result = true_fn()
            for scope in reversed(Module.scope_stack):
                scope.__exit__(None, None, None)
            return result

        def false_fn_wrapper():
            for scope in Module.scope_stack:
                scope.__enter__()
            result = false_fn()
            for scope in reversed(Module.scope_stack):
                scope.__exit__(None, None, None)
            return result

        Module.cond_counter += 1
        x = tf.cond(pred=pred, true_fn=true_fn_wrapper, false_fn=false_fn_wrapper)
        Module.cond_counter -= 1
        return x

    def while_loop(
        self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10,
        back_prop=False, swap_memory=False, maximum_iterations=None
    ):
        Module.while_counter += 1
        if maximum_iterations is not None and maximum_iterations.dtype is not tf.int32:
            maximum_iterations = tf.dtypes.cast(x=maximum_iterations, dtype=tf.int32)
        x = tf.while_loop(
            cond=cond, body=body, loop_vars=loop_vars, shape_invariants=shape_invariants,
            parallel_iterations=parallel_iterations, back_prop=back_prop,
            swap_memory=swap_memory, maximum_iterations=maximum_iterations
        )
        Module.while_counter -= 1
        return x

    def add_variable(
        self, name, dtype, shape, is_trainable, initializer='zeros', is_saved=True, summarize=None,
        shared=None
    ):
        # name
        if not util.is_valid_name(name=name):
            raise TensorforceError.value(name='variable', argument='name', value=name)
        elif name in self.variables:
            raise TensorforceError.exists(name='variable', value=name)
        # dtype
        if not util.is_valid_type(dtype=dtype):
            raise TensorforceError.value(name='variable', argument='dtype', value=dtype)
        # shape
        if not util.is_iterable(x=shape) or not all(isinstance(dims, int) for dims in shape):
            raise TensorforceError.type(name='variable', argument='shape', value=shape)
        elif not all(dims > 0 for dims in shape):
            raise TensorforceError.value(name='variable', argument='shape', value=shape)
        # is_trainable
        if not isinstance(is_trainable, bool):
            raise TensorforceError.type(
                name='variable', argument='is_trainable', value=is_trainable
            )
        elif is_trainable and dtype != 'float':
            raise TensorforceError.unexpected()
        # initializer
        initializer_names = (
            'normal', 'normal-relu', 'orthogonal', 'orthogonal-relu', 'zeros', 'ones'
        )
        if not isinstance(initializer, (util.py_dtype(dtype=dtype), np.ndarray, tf.Tensor)) and \
                initializer not in initializer_names:
            raise TensorforceError.value(
                name='variable', argument='initializer', value=initializer
            )
        elif isinstance(initializer, np.ndarray) and \
                initializer.dtype != util.np_dtype(dtype=dtype):
            raise TensorforceError.type(
                name='variable', argument='initializer', value=initializer
            )
        elif isinstance(initializer, tf.Tensor) and util.dtype(x=initializer) != dtype:
            raise TensorforceError.type(
                name='variable', argument='initializer', value=initializer
            )
        # is_saved
        if not isinstance(is_saved, bool):
            raise TensorforceError.type(name='variable', argument='is_saved', value=is_saved)
        # summarize
        if summarize is not None and not isinstance(summarize, bool):
            raise TensorforceError.type(name='variable', argument='summarize', value=summarize)
        # shared
        if shared is not None and not isinstance(shared, str):
            raise TensorforceError.type(name='variable', argument='shared', value=shared)

        variable = None

        if shared is not None and len(self.graph.get_collection(name=shared)) > 0:
            # Retrieve shared variable from TensorFlow
            collection = self.graph.get_collection(name=shared)
            if len(collection) > 1:
                raise TensorforceError.unexpected()
            variable = collection[0]

        else:
            tf_dtype = util.tf_dtype(dtype=dtype)

            # Variable initializer
            if isinstance(initializer, util.py_dtype(dtype=dtype)):
                initializer = tf.constant(value=initializer, dtype=tf_dtype, shape=shape)
            elif isinstance(initializer, np.ndarray):
                if initializer.shape != shape:
                    raise TensorforceError(
                        "Invalid variable initializer shape: {}.".format(initializer.shape)
                    )
                initializer = tf.constant(value=initializer, dtype=tf_dtype)
            elif isinstance(initializer, tf.Tensor):
                if util.shape(x=initializer) != shape:
                    raise TensorforceError(
                        "Invalid variable initializer shape: {}.".format(util.shape(x=initializer))
                    )
                initializer = initializer
            elif not isinstance(initializer, str):
                raise TensorforceError("Invalid variable initializer: {}".format(initializer))
            elif initializer[:6] == 'normal':
                if dtype != 'float':
                    raise TensorforceError(
                        message="Invalid variable initializer value for non-float variable: {}.".format(
                            initializer
                        )
                    )
                if initializer[6:] == '-relu':
                    stddev = min(0.1, sqrt(2.0 / util.product(xs=shape[:-1])))
                else:
                    stddev = min(0.1, sqrt(2.0 / (util.product(xs=shape[:-1]) + shape[-1])))
                initializer = tf.random.normal(shape=shape, stddev=stddev, dtype=tf_dtype)
            elif initializer[:10] == 'orthogonal':
                if dtype != 'float':
                    raise TensorforceError(
                        message="Invalid variable initializer value for non-float variable: {}.".format(
                            initializer
                        )
                    )
                if len(shape) < 2:
                    raise TensorforceError(
                        message="Invalid variable initializer value for 0/1-rank variable: {}.".format(
                            initializer
                        )
                    )
                normal = np.random.normal(size=(util.product(xs=shape[:-1]), shape[-1]))
                u, _, v = np.linalg.svd(a=normal, full_matrices=False)
                orthogonal = u if u.shape[1] == shape[-1] else v
                if initializer[10:] == '-relu':
                    orthogonal = orthogonal * sqrt(2.0)
                initializer = tf.constant(value=orthogonal.reshape(shape), dtype=tf_dtype)
            elif initializer == 'zeros':
                initializer = tf.zeros(shape=shape, dtype=tf_dtype)
            elif initializer == 'ones':
                initializer = tf.ones(shape=shape, dtype=tf_dtype)

            # Variable
            variable = tf.Variable(
                initial_value=initializer, trainable=is_trainable, validate_shape=True, name=name,
                dtype=tf_dtype, shape=shape
            )

            # Register shared variable with TensorFlow
            if shared is not None:
                self.graph.add_to_collection(name=shared, value=variable)

        # Register variable
        self.variables[name] = variable
        if is_trainable:
            self.trainable_variables[name] = variable
        if is_saved:
            self.saved_variables[name] = variable

        # Add summary
        if (summarize is None and is_trainable) or summarize:
            variable = self.add_summary(
                label='variables', name=name, tensor=variable, mean_variance=True
            )
            variable = self.add_summary(label='variables-histogram', name=name, tensor=variable)

        return variable

    def add_placeholder(self, name, dtype, shape, batched, default=None):
        # name
        name = name + '-input'
        if not util.is_valid_name(name=name):
            raise TensorforceError.value(name='placeholder', argument='name', value=name)
        # dtype
        if not util.is_valid_type(dtype=dtype):
            raise TensorforceError.value(name='placeholder', argument='dtype', value=dtype)
        # shape
        if not util.is_iterable(x=shape) or not all(isinstance(dims, int) for dims in shape):
            raise TensorforceError.type(name='placeholder', argument='shape', value=shape)
        elif not all(dims > 0 for dims in shape):
            raise TensorforceError.value(name='placeholder', argument='shape', value=shape)
        # batched
        if not isinstance(batched, bool):
            raise TensorforceError.type(name='placeholder', argument='batched', value=batched)
        # default
        if default is not None:
            # if batched:
            #     raise TensorforceError.unexpected()
            if not isinstance(default, tf.Tensor):
                raise TensorforceError.unexpected()
            elif util.dtype(x=default) != dtype:
                raise TensorforceError.unexpected()

        # Placeholder
        if batched:
            shape = (None,) + shape
        if default is None:
            dtype = util.tf_dtype(dtype=dtype)
            placeholder = tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
        else:
            # check dtype and shape !!!
            placeholder = tf.compat.v1.placeholder_with_default(
                input=default, shape=shape, name=name
            )

        return placeholder

    def is_summary_logged(self, label):
        # Check whether any summaries are logged
        if self.summary_labels is None:
            return False

        # Check whether not in while loop
        if Module.while_counter > 0:
            return False
        # Check whether not in nested condition
        if Module.cond_counter > 1:
            return False

        # Temporary
        if label == 'variables' or label == 'variables-histogram':
            return False

        # Check whether given label is logged
        if util.is_iterable(x=label):
            assert all(not x.endswith('-histogram') for x in label)
            if self.summary_labels != 'all' and all(x not in self.summary_labels for x in label):
                return False
        else:
            if (self.summary_labels != 'all' or label.endswith('-histogram')) and \
                    label not in self.summary_labels:
                return False

        return True

    def add_summary(
        self, label, name, tensor, pass_tensors=None, return_summaries=False, mean_variance=False,
        enumerate_last_rank=False
    ):
        # should be "labels" !!!
        # label
        if util.is_iterable(x=label):
            if not all(isinstance(x, str) for x in label):
                raise TensorforceError.type(name='summary', argument='label', value=label)
        else:
            if not isinstance(label, str):
                raise TensorforceError.type(name='summary', argument='label', value=label)
        # name
        if not isinstance(name, str):
            raise TensorforceError.type(name='summary', argument='name', value=name)
        # tensor
        if not isinstance(tensor, (tf.Tensor, tf.Variable)):
            raise TensorforceError.type(name='summary', argument='tensor', value=tensor)
        # pass_tensors
        if util.is_iterable(x=pass_tensors):
            if not all(isinstance(x, (tf.Tensor, tf.IndexedSlices)) for x in pass_tensors):
                raise TensorforceError.type(
                    name='summary', argument='pass_tensors', value=pass_tensors
                )
        elif pass_tensors is not None:
            if not isinstance(pass_tensors, tf.Tensor):
                raise TensorforceError.type(
                    name='summary', argument='pass_tensors', value=pass_tensors
                )
        # enumerate_last_rank
        if not isinstance(enumerate_last_rank, bool):
            raise TensorforceError.type(
                name='summary', argument='enumerate_last_rank', value=tensor
            )

        if pass_tensors is None:
            pass_tensors = tensor

        # Check whether summary is logged
        if not self.is_summary_logged(label=label):
            return pass_tensors

        # Add to available summaries
        if util.is_iterable(x=label):
            self.available_summaries.update(label)
        else:
            self.available_summaries.add(label)

        # Handle enumerate_last_rank
        if enumerate_last_rank:
            dims = util.shape(x=tensor)[-1]
            tensors = OrderedDict([(name + str(n), tensor[..., n]) for n in range(dims)])
        else:
            tensors = OrderedDict([(name, tensor)])

        if mean_variance:
            for name in list(tensors):
                tensor = tensors.pop(name)
                mean, variance = tf.nn.moments(x=tensor, axes=tuple(range(util.rank(x=tensor))))
                tensors[name + '-mean'] = mean
                tensors[name + '-variance'] = variance

        # Scope handling
        if Module.scope_stack is not None:
            for scope in reversed(Module.scope_stack[1:]):
                scope.__exit__(None, None, None)
            if len(Module.global_scope) > 0:
                temp_scope = tf.name_scope(name='/'.join(Module.global_scope))
                temp_scope.__enter__()
            tensors = util.fmap(function=util.identity_operation, xs=tensors)

        # TensorFlow summaries
        assert Module.global_summary_step is not None
        step = Module.retrieve_tensor(name=Module.global_summary_step)
        summaries = list()
        for name, tensor in tensors.items():
            shape = util.shape(x=tensor)
            if shape == ():
                summaries.append(tf.summary.scalar(name=name, data=tensor, step=step))
            elif shape == (-1,):
                tensor = tf.math.reduce_sum(input_tensor=tensor, axis=0)
                summaries.append(tf.summary.scalar(name=name, data=tensor, step=step))
            elif shape == (1,):
                tensor = tf.squeeze(input=tensor, axis=-1)
                summaries.append(tf.summary.scalar(name=name, data=tensor, step=step))
            elif shape == (-1, 1):
                tensor = tf.math.reduce_sum(input_tensor=tf.squeeze(input=tensor, axis=-1), axis=0)
                summaries.append(tf.summary.scalar(name=name, data=tensor, step=step))
            else:
                # General tensor as histogram
                assert not util.is_iterable(x=label) and label.endswith('-histogram')
                summaries.append(tf.summary.histogram(name=name, data=tensor, step=step))

        # Scope handling
        if Module.scope_stack is not None:
            if len(Module.global_scope) > 0:
                temp_scope.__exit__(None, None, None)
            for scope in Module.scope_stack[1:]:
                scope.__enter__()

        with tf.control_dependencies(control_inputs=summaries):
            return util.fmap(function=util.identity_operation, xs=pass_tensors)

    @staticmethod
    def get_module_class_and_kwargs(
        name, module=None, modules=None, default_module=None, **kwargs
    ):
        # name
        if not util.is_valid_name(name=name):
            raise TensorforceError.value(name='module', argument='name', value=name)
        # module
        # ???
        # modules
        if modules is not None and not isinstance(modules, dict):
            raise TensorforceError.type(name='module', argument='modules', value=modules)
        # default_module
        # ???
        if isinstance(module, dict):
            # Dictionary module specification (type either given via 'type' or 'default_module')
            util.deep_disjoint_update(target=kwargs, source=module)
            module = kwargs.pop('type', default_module)
            return Module.get_module_class_and_kwargs(
                name=name, module=module, modules=modules, default_module=default_module, **kwargs
            )

        elif isinstance(module, str):
            if os.path.isfile(module):
                # JSON file module specification
                with open(module, 'r') as fp:
                    module = json.load(fp=fp)
                return Module.get_module_class_and_kwargs(
                    name=name, module=module, modules=modules, default_module=default_module,
                    **kwargs
                )

            elif '.' in module:
                # Library module specification
                library_name, module_name = module.rsplit('.', 1)
                library = importlib.import_module(name=library_name)
                module = getattr(library, module_name)
                return Module.get_module_class_and_kwargs(
                    name=name, module=module, modules=modules, default_module=default_module,
                    **kwargs
                )

            elif modules is not None and module in modules:
                # Keyword module specification
                return Module.get_module_class_and_kwargs(
                    name=name, module=modules[module], default_module=default_module, **kwargs
                )

            elif 'default' in modules or default_module is not None:
                # Default module specification
                if '_first_arg' in kwargs:
                    raise TensorforceError.value(name='module kwargs', value='_first_arg')
                if module is not None:
                    kwargs['_first_arg'] = module
                if default_module is None:
                    default_module = modules['default']
                return Module.get_module_class_and_kwargs(
                    name=name, module=default_module, modules=modules, **kwargs
                )

            else:
                raise TensorforceError.value(name='module specification', value=module)

        elif not callable(module) and ('default' in modules or default_module is not None):
            # Default module specification
            if '_first_arg' in kwargs:
                raise TensorforceError.value(name='module kwargs', value='_first_arg')
            if module is not None:
                kwargs['_first_arg'] = module
            if default_module is None:
                default_module = modules['default']
            return Module.get_module_class_and_kwargs(
                name=name, module=default_module, modules=modules, **kwargs
            )

        elif callable(module):
            # for key, arg in kwargs.items():
            #     assert arg is not None, (key, arg)
            #     if arg is None:
            #         assert False
            #         kwargs.pop(key)
            first_arg = kwargs.pop('_first_arg', None)
            return module, first_arg, kwargs

        else:
            raise TensorforceError.value(name='module specification', value=module)

    def add_module(
        self, name, module=None, modules=None, default_module=None, is_trainable=True,
        is_saved=True, is_subscope=False, **kwargs
    ):
        # name
        if name in self.modules:
            raise TensorforceError.exists(name='sub-module', value=name)
        # is_trainable
        if not isinstance(is_trainable, bool):
            raise TensorforceError.type(name='module', argument='is_trainable', value=is_trainable)
        # is_saved
        if not isinstance(is_saved, bool):
            raise TensorforceError.type(name='module', argument='is_saved', value=is_saved)

        module_cls, first_arg, kwargs = Module.get_module_class_and_kwargs(
            name=name, module=module, modules=modules, default_module=default_module, **kwargs
        )

        # Final callable module specification
        if Module.global_scope is None:
            raise TensorforceError.unexpected()

        # Global scope handling
        Module.is_add_module = True
        if is_subscope:
            Module.global_scope.append(name)

        # Set internal attributes
        Module.set_parent = self

        # Inherit arguments
        Module.inherit_l2_regularization = self.l2_regularization
        Module.inherit_summary_labels = self.summary_labels

        # Module constructor
        if first_arg is None:
            module = module_cls(name, **kwargs)
        else:
            module = module_cls(name, first_arg, **kwargs)

        # Reset
        Module.set_parent = None
        Module.inherit_l2_regularization = None
        Module.inherit_summary_labels = None

        # Global scope handling
        if is_subscope:
            Module.global_scope.pop()
        Module.is_add_module = False

        # Internal attributes
        module.is_subscope = is_subscope

        # Register module
        self.modules[name] = module
        if is_trainable:
            self.trainable_modules[name] = module
        if is_saved:
            self.saved_modules[name] = module

        return module

    def get_variables(self, only_trainable=False, only_saved=False):
        # only_trainable
        if not isinstance(only_trainable, bool):
            raise TensorforceError.type(name='get_variables', argument='only_trainable', value=only_trainable)
        # only_saved
        if not isinstance(only_saved, bool):
            raise TensorforceError.type(name='get_variables', argument='only_saved', value=only_saved)
        # not both
        if only_trainable and only_saved:
            raise TensorforceError.unexpected()

        if only_trainable:
            # Only trainable variables
            variables = list(self.trainable_variables.values())
            for module in self.trainable_modules.values():
                variables.extend(module.get_variables(only_trainable=only_trainable))

        elif only_saved:
            # Only saved variables
            variables = list(self.saved_variables.values())
            for module in self.saved_modules.values():
                variables.extend(module.get_variables(only_saved=only_saved))

        else:
            # All variables
            variables = list(self.variables.values())
            for module in self.modules.values():
                variables.extend(module.get_variables(only_trainable=only_trainable))

        return variables

    def get_available_summaries(self):
        summaries = set(self.available_summaries)
        for module in self.modules.values():
            summaries.update(module.get_available_summaries())
        return sorted(summaries)
