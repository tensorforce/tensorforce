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
from functools import partial
import importlib
import json
from math import sqrt
import os
import time

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
import tensorforce.core


MODULE_STACK = list()

def get_global_scope():
    for n, module in enumerate(MODULE_STACK):
        assert isinstance(module, (Module, tf.name_scope))
        yield module.name
        # elif n == 1 and isinstance(module, str):
        #     yield module
        # else:
        #     raise TensorforceError.unexpected()




def tf_function(num_args):

    def decorator(function):

        def decorated(self, *args, **kwargs):
            if len(kwargs) > 0:
                assert len(args) == 0 and len(kwargs) >= num_args
                all_args = list(kwargs.values())
                is_args = False
            else:
                assert len(kwargs) == 0 and len(args) == num_args
                all_args = list(args)
                is_args = True
            name = function.__name__
            assert function.__qualname__.endswith('.' + name)

            if not hasattr(self, name + '_graphs'):
                setattr(self, name + '_graphs', OrderedDict())

            function_graphs = getattr(self, name + '_graphs')
            # graph_signature = (function.__qualname__,) + tuple(all_args[num_args:])
            graph_signature = tuple(all_args[num_args:])
            graph_input = util.to_list(xs=all_args[:num_args])

            if graph_signature not in function_graphs:
                assert len(function_graphs) == 0 or \
                    len(next(iter(function_graphs))) == len(graph_signature)

                input_signature = self.input_signature(function=name)
                assert len(input_signature) == len(graph_input)
                signature_kwargs = dict(list(kwargs.items())[num_args:])

                def function_graph(*args):
                    if self not in MODULE_STACK:
                        MODULE_STACK.append(self)
                        pop_module_stack = True
                    else:
                        pop_module_stack = False
                    if self.device is not None:
                        self.device.__enter__()
                    # with self.name_scope:
                    if is_args:
                        results = Module.with_name_scope(method=function)(
                            self, *args, **signature_kwargs
                        )
                    else:
                        graph_kwargs = util.from_list(xs=args, ys=kwargs)
                        results = Module.with_name_scope(method=function)(
                            self, **graph_kwargs, **signature_kwargs
                        )
                    if self.device is not None:
                        self.device.__exit__(None, None, None)
                    if pop_module_stack:
                        popped = MODULE_STACK.pop()
                        assert popped is self
                    return results

                # function_graph = partial(function, self, **graph_kwargs)
                function_graphs[graph_signature] = (function.__qualname__, tf.function(
                    func=function_graph, input_signature=input_signature, autograph=False
                    # experimental_implements=None, experimental_autograph_options=None,
                    # experimental_relax_shapes=False, experimental_compile=None
                ))

            qualname, function_graph = function_graphs[graph_signature]

            if function.__qualname__ != qualname:
                return function(self, *args, **kwargs)
            else:
                return function_graph(*graph_input)

        return tf.compat.v1.flags.tf_decorator.make_decorator(
            target=function, decorator_func=decorated
        )

    return decorator


class Module(tf.Module):
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

    _TF_MODULE_IGNORED_PROPERTIES = frozenset((
        '_self_unconditional_checkpoint_dependencies',
        '_self_unconditional_dependency_names',
        'parent'
    ))

    global_summary_step = None

    def __init__(
        self, name, is_root=False, device=None, summary_labels=None, l2_regularization=None
    ):
        if not util.is_valid_name(name=name):
            raise TensorforceError.value(name='module', argument='name', value=name)

        super().__init__(name=name)

        self.is_root = is_root
        self.device = device
        self.is_initialized = False

        if self.is_root:
            MODULE_STACK.clear()
            self.is_trainable = True
            self.is_saved = True
            self.global_tensors_spec = OrderedDict()
            self.input_tensors = None
            self.output_tensors = None
            self.query_tensors = None
            self.available_summaries = None
        else:
            assert len(MODULE_STACK) > 0
            self.parent = MODULE_STACK[-1]
            self.is_trainable = None
            self.is_saved = None

        MODULE_STACK.append(self)

        if self.is_root and summary_labels is None:
            self.summary_labels = set()
        elif summary_labels is None or summary_labels == 'all':
            self.summary_labels = summary_labels
        elif not all(isinstance(label, str) for label in summary_labels):
            raise TensorforceError.value(
                name='module', argument='summary_labels', value=summary_labels
            )
        else:
            self.summary_labels = set(summary_labels)

        if self.is_root and l2_regularization is None:
            l2_regularization = 0.0
        if l2_regularization is None:
            self.l2_regularization = None 
        else:
            self.l2_regularization = self.add_module(
                name='l2_regularization', module=l2_regularization,
                modules=tensorforce.core.parameter_modules, is_trainable=False, dtype='float',
                min_value=0.0
            )

    @property
    def root(self):
        module = self
        while not module.is_root:
            module = module.parent
        return module

    def tf_initialize(self):
        pass

    @tf_function(num_args=0)
    def regularize(self):
        zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))

        module = self
        while module.l2_regularization is None:
            module = module.parent

        if len(self.this_trainable_variables) == 0 or module.l2_regularization.max_value() == 0.0:
            regularization_loss = zero

        else:
            l2_regularization = module.l2_regularization.value()

            def no_l2_regularization():
                return zero

            def apply_l2_regularization():
                l2_variables = list()
                for variable in self.this_trainable_variables:
                    if variable.dtype != util.tf_dtype(dtype='float'):
                        variable = tf.dtypes.cast(x=variable, dtype=util.tf_dtype(dtype='float'))
                    l2_variables.append(tf.reduce_sum(input_tensor=tf.square(x=variable)))
                return l2_regularization * tf.math.add_n(inputs=l2_variables)

            skip_l2_regularization = tf.math.equal(x=l2_regularization, y=zero)
            regularization_loss = self.cond(
                pred=skip_l2_regularization, true_fn=no_l2_regularization,
                false_fn=apply_l2_regularization
            )

        for module in self.this_submodules:
            if getattr(module, 'is_trainable', True):
                regularization_loss += module.regularize()

        return regularization_loss

    # @tf.Module.with_name_scope
    def initialize(self):
        # Check whether module is already initialized
        if self.is_initialized:
            return
            raise TensorforceError(message=("Module is already initialized: " + str(self)))

        # Set internal attributes
        self.input_tensors = OrderedDict()
        self.output_tensors = OrderedDict()
        self.query_tensors = OrderedDict()
        self.available_summaries = set()

        if self.is_root:
            popped = MODULE_STACK.pop()
            assert popped is self

        MODULE_STACK.append(self)

        # TensorFlow device
        if self.device is not None:
            self.device = tf.device(device_name_or_function=self.device)
            self.device.__enter__()

        with self.name_scope:

            if self.is_root:
                # Timestep counter
                self.timesteps = self.add_variable(
                    name='timesteps', dtype='long', shape=(), is_trainable=False,
                    initializer='zeros', is_global=True
                )
                collection = self.graph.get_collection(name='global_step')
                assert len(collection) == 0
                self.graph.add_to_collection(name='global_step', value=self.timesteps)

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

                    Module.global_summary_step = 'timesteps'
                    condition = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
                    record_summaries = tf.summary.record_if(condition=condition)
                    record_summaries.__enter__()

                # Assignment values
                self.assignment_input = dict(
                    bool=self.add_placeholder(
                        name='assignment-bool', dtype='bool', shape=None, batched=False
                    ), int=self.add_placeholder(
                        name='assignment-int', dtype='int', shape=None, batched=False
                    ), long=self.add_placeholder(
                        name='assignment-long', dtype='long', shape=None, batched=False
                    ), float=self.add_placeholder(
                        name='assignment-float', dtype='float', shape=None, batched=False
                    )
                )

                # Delayed global-timestep assign operation
                self.timesteps.assign(value=self.assignment_input['long'], name='timestep-assign')

                # Episode counter
                self.episodes = self.add_variable(
                    name='episodes', dtype='long', shape=(), is_trainable=False,
                    initializer='zeros', is_global=True
                )

                # Update counter
                self.updates = self.add_variable(
                    name='updates', dtype='long', shape=(), is_trainable=False, initializer='zeros',
                    is_global=True
                )

                if self.summarizer_spec is not None:
                    if len(self.summarizer_spec.get('custom', ())) > 0:
                        self.summarize_input = self.add_placeholder(
                            name='summarize', dtype='float', shape=None, batched=False
                        )
                        # self.summarize_step_input = self.add_placeholder(
                        #     name='summarize-step', dtype='long', shape=(), batched=False,
                        #     default=self.timesteps
                        # )
                        self.summarize_step_input = self.timesteps
                        self.custom_summaries = OrderedDict()
                        for name, summary in self.summarizer_spec['custom'].items():
                            if summary['type'] == 'audio':
                                self.custom_summaries[name] = tf.summary.audio(
                                    name=name, data=self.summarize_input,
                                    sample_rate=summary['sample_rate'],
                                    step=self.summarize_step_input,
                                    max_outputs=summary.get('max_outputs', 3),
                                    encoding=summary.get('encoding')
                                )
                            elif summary['type'] == 'histogram':
                                self.custom_summaries[name] = tf.summary.histogram(
                                    name=name, data=self.summarize_input,
                                    step=self.summarize_step_input,
                                    buckets=summary.get('buckets')
                                )
                            elif summary['type'] == 'image':
                                self.custom_summaries[name] = tf.summary.image(
                                    name=name, data=self.summarize_input,
                                    step=self.summarize_step_input,
                                    max_outputs=summary.get('max_outputs', 3)
                                )
                            elif summary['type'] == 'scalar':
                                self.custom_summaries[name] = tf.summary.scalar(
                                    name=name,
                                    data=tf.reshape(tensor=self.summarize_input, shape=()),
                                    step=self.summarize_step_input
                                )
                            else:
                                raise TensorforceError.value(
                                    name='custom summary', argument='type', value=summary['type'],
                                    hint='not in {audio,histogram,image,scalar}'
                                )

                    record_summaries.__exit__(None, None, None)

                    Module.global_summary_step = 'updates'
                    if 'frequency' not in self.summarizer_spec or \
                            isinstance(self.summarizer_spec['frequency'], int):
                        condition = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))

                    elif 'variables' in self.summarizer_spec['frequency']:
                        step = self.global_tensor(name=Module.global_summary_step)
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

            for module in self.this_submodules:
                if isinstance(module, Module):
                    module.initialize()
            self.tf_initialize()

        self.is_initialized = True

        if self.is_root and self.summarizer_spec is not None:
            record_summaries.__exit__(None, None, None)
            Module.global_summary_step = None

        if self.device is not None:
            self.device.__exit__(None, None, None)

        popped = MODULE_STACK.pop()
        assert popped is self

        # Internal TensorFlow functions, prefixed by 'tf_'
        for attribute in sorted(dir(self)):
            if attribute.startswith('tf_') and attribute != 'tf_initialize':
                assert False

        #  API TensorFlow functions, prefixed by 'api_'
        for attribute in sorted(dir(self)):
            if attribute.startswith('api_'):
                function_name = attribute[4:]
                assert hasattr(self, 'config')
                if self.config is not None and 'api_functions' in self.config and \
                        function_name not in self.config['api_functions']:
                    continue

                if function_name in ('act', 'independent_act'):
                    Module.global_summary_step = 'timesteps'
                elif function_name in ('observe', 'experience'):
                    Module.global_summary_step = 'episodes'
                elif function_name == 'update':
                    Module.global_summary_step = 'updates'

                if self.summarizer_spec is not None:
                    if 'frequency' not in self.summarizer_spec:
                        condition = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))

                    elif isinstance(self.summarizer_spec['frequency'], int):
                        if function_name in ('act', 'independent_act'):
                            step = self.timesteps
                            frequency = tf.constant(
                                value=self.summarizer_spec['frequency'],
                                dtype=util.tf_dtype(dtype='long')
                            )
                            zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
                            condition = (
                                lambda: tf.math.equal(x=tf.math.mod(x=step, y=frequency), y=zero)
                            )
                        elif function_name in ('reset', 'independent_act'):
                            condition = tf.constant(value=False, dtype=util.tf_dtype(dtype='bool'))
                        else:
                            condition = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))

                    elif function_name in self.summarizer_spec['frequency']:
                        if function_name in ('act', 'independent_act'):
                            step = self.timesteps
                        elif function_name in ('observe', 'experience'):
                            step = self.episodes
                        elif function_name == 'update':
                            step = self.updates
                        elif function_name == 'reset':
                            raise TensorforceError.value(
                                name='module', argument='summarizer[frequency]',
                                value=function_name,
                                hint='not in {act,experience,observe,update}'
                            )
                        else:
                            raise TensorforceError.value(
                                name='module', argument='summarizer[frequency]',
                                value=function_name,
                                hint='not in {act,experience,observe,update}'
                            )
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
                    raise TensorforceError.unexpected()
                if hasattr(self, function_name):
                    raise TensorforceError.unexpected()

                api_function = getattr(self, attribute)
                if not callable(api_function):
                    raise TensorforceError.unexpected()

                function = self.create_api_function(
                    name='{}.{}'.format(self.name, function_name), api_function=api_function
                )

                setattr(self, function_name, function)

                if self.summarizer_spec is not None:
                    record_summaries.__exit__(None, None, None)
                    Module.global_summary_step = None

        if self.is_root:
            self.graph_summary = None  # TODO!
            if self.summarizer_spec is not None:
                default_summarizer.__exit__(None, None, None)

    def input_signature(self, function):
        if function == 'regularize':
            return ()

        else:
            return None

    def create_api_function(self, name, api_function):
        # Call API TensorFlow function
        MODULE_STACK.append(self)
        MODULE_STACK.append(tf.name_scope(name=name[name.index('.') + 1:]))

        if self.device is not None:
            self.device.__enter__()
        scope = tf.name_scope(name=name)
        # Module.scope_stack.append(scope)
        scope.__enter__()

        results = api_function()
        assert all(x.name.endswith('-output:0') for x in util.flatten(xs=results))
        self.output_tensors[name[name.index('.') + 1:]] = [
            x.name[len(name) + 1: -9] for x in util.flatten(xs=results)
        ]

        # Function-level identity operation for retrieval
        query_tensors = set()
        # for scoped_name, tensor in Module.queryable_tensors.items():
        #     tensor = util.identity_operation(x=tensor, operation_name=(scoped_name + '-query'))
        #     assert tensor.name.endswith('-query:0')
        #     assert scoped_name not in query_tensors
        #     query_tensors.add(scoped_name)
        for scoped_name in self.global_tensors_spec:
            scoped_name1 = scoped_name.replace('agent/', '')
            scoped_name2 = scoped_name.replace('agent/', name.replace('.', '/') + '/')
            collection = self.root.graph.get_collection(name=scoped_name2)
            if len(collection) == 0:
                continue
            tensor = util.identity_operation(x=collection[0], operation_name=(scoped_name1 + '-query'))
            assert tensor.name.endswith('-query:0')
            assert scoped_name not in query_tensors
            query_tensors.add(scoped_name)
        self.query_tensors[name[name.index('.') + 1:]] = sorted(query_tensors)

        scope.__exit__(None, None, None)
        if self.device is not None:
            self.device.__exit__(None, None, None)

        MODULE_STACK.pop()
        popped = MODULE_STACK.pop()
        assert popped is self

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
                raise TensorforceError.value(
                    name=api_function, argument='inputs', value=list(feed_dict)
                )

            # Fetches value/tuple
            fetches = util.fmap(function=(lambda x: x.name), xs=results)
            if query is not None:
                # If additional tensors are to be fetched
                query = util.fmap(
                    function=(lambda x: util.join_scopes(name, x) + '-query:0'), xs=query
                )
                if util.is_iterable(x=fetches):
                    fetches = tuple(fetches) + (query,)
                else:
                    fetches = (fetches, query)
            if not util.reduce_all(
                predicate=(lambda x: x.endswith('-output:0') or x.endswith('-query:0')), xs=fetches
            ):
                raise TensorforceError.value(
                    name=api_function, argument='outputs', value=list(fetches)
                )

            # TensorFlow session call
            fetched = self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)

            return fetched

        return fn

    def cond(self, pred, true_fn, false_fn):

        # def true_fn_wrapper():
        #     for scope in Module.scope_stack:
        #         scope.__enter__()
        #     result = true_fn()
        #     for scope in reversed(Module.scope_stack):
        #         scope.__exit__(None, None, None)
        #     return result

        # def false_fn_wrapper():
        #     for scope in Module.scope_stack:
        #         scope.__enter__()
        #     result = false_fn()
        #     for scope in reversed(Module.scope_stack):
        #         scope.__exit__(None, None, None)
        #     return result

        # Module.cond_counter += 1
        x = tf.cond(pred=pred, true_fn=true_fn, false_fn=false_fn)
        # Module.cond_counter -= 1
        return x

    def while_loop(
        self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10,
        back_prop=False, swap_memory=False, maximum_iterations=None
    ):
        # Module.while_counter += 1
        if maximum_iterations is not None and maximum_iterations.dtype is not tf.int32:
            maximum_iterations = tf.dtypes.cast(x=maximum_iterations, dtype=tf.int32)
        x = tf.while_loop(
            cond=cond, body=body, loop_vars=loop_vars, shape_invariants=shape_invariants,
            parallel_iterations=parallel_iterations, back_prop=back_prop,
            swap_memory=swap_memory, maximum_iterations=maximum_iterations
        )
        # Module.while_counter -= 1
        return x

    def set_global_tensor(self, name, tensor):
        assert self.is_initialized

        if not isinstance(tensor, tf.Tensor):
            raise TensorforceError.unexpected()

        spec = dict(type=util.dtype(x=tensor), shape=util.shape(x=tensor))
        scoped_name = util.join_scopes(*get_global_scope(), name)
        if scoped_name in self.root.global_tensors_spec:
            assert self.root.global_tensors_spec[scoped_name] == spec
        else:
            self.root.global_tensors_spec[scoped_name] = spec

        collection = self.root.graph.get_collection_ref(name=scoped_name)
        if len(collection) > 1:
            raise TensorforceError.unexpected()
        elif len(collection) > 0:
            collection[0] = tensor
        else:
            self.root.graph.add_to_collection(name=scoped_name, value=tensor)

    def global_tensor(self, name):
        assert self.is_initialized

        global_scope = list(get_global_scope())
        for n in range(len(global_scope), 0, -1):
            scoped_name = util.join_scopes(*global_scope[:n], name)
            if scoped_name in self.root.global_tensors_spec:
                break
        else:
            raise TensorforceError.unexpected()

        collection = self.root.graph.get_collection(name=scoped_name)
        if len(collection) != 1:
            raise TensorforceError.unexpected()

        return collection[0]

    def add_variable(
        self, name, dtype, shape, is_trainable, initializer='zeros', is_saved=True, is_global=False,
        summarize=None
    ):
        assert not self.is_initialized
        # name
        if not util.is_valid_name(name=name):
            raise TensorforceError.value(name='Module.add_variable', argument='name', value=name)
        # elif name in self.variables:
        #     raise TensorforceError.exists(name='variable', value=name)
        # dtype
        if not util.is_valid_type(dtype=dtype):
            raise TensorforceError.value(name='Module.add_variable', argument='dtype', value=dtype)
        # shape
        if not util.is_iterable(x=shape) or not all(isinstance(dims, int) for dims in shape):
            raise TensorforceError.value(name='Module.add_variable', argument='shape', value=shape)
        elif not all(dims > 0 for dims in shape):
            raise TensorforceError.value(name='Module.add_variable', argument='shape', value=shape)
        # is_trainable
        if not isinstance(is_trainable, bool):
            raise TensorforceError.type(
                name='Module.add_variable', argument='is_trainable', dtype=type(is_trainable)
            )
        elif is_trainable and dtype != 'float':
            raise TensorforceError.value(
                name='Module.add_variable', argument='is_trainable', value=is_trainable,
                condition='dtype != float'
            )
        # initializer
        initializer_names = (
            'normal', 'normal-relu', 'orthogonal', 'orthogonal-relu', 'zeros', 'ones'
        )
        if not isinstance(initializer, (util.py_dtype(dtype=dtype), np.ndarray, tf.Tensor)) and \
                initializer not in initializer_names:
            raise TensorforceError.value(
                name='Module.add_variable', argument='initializer', value=initializer
            )
        elif isinstance(initializer, np.ndarray) and \
                initializer.dtype != util.np_dtype(dtype=dtype):
            raise TensorforceError.type(
                name='Module.add_variable', argument='initializer', dtype=type(initializer)
            )
        elif isinstance(initializer, tf.Tensor) and util.dtype(x=initializer) != dtype:
            raise TensorforceError.type(
                name='Module.add_variable', argument='initializer', dtype=type(initializer)
            )
        # is_saved
        if not isinstance(is_saved, bool):
            raise TensorforceError.type(
                name='Module.add_variable', argument='is_saved', dtype=type(is_saved)
            )
        # is_global
        if not isinstance(is_global, bool):
            raise TensorforceError.type(
                name='Module.add_variable', argument='is_global', dtype=type(is_global)
            )
        # summarize
        if summarize is not None and not isinstance(summarize, bool):
            raise TensorforceError.type(
                name='Module.add_variable', argument='summarize', dtype=type(summarize)
            )

        if is_global and len(self.graph.get_collection(name=(name + '-variable'))) > 0:
            # Retrieve global variable from TensorFlow
            collection = self.graph.get_collection(name=name)
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
                    raise TensorforceError.mismatch(
                        name='Module.add_variable', value1='shape', value2='initializer'
                    )
                initializer = tf.constant(value=initializer, dtype=tf_dtype)
            elif isinstance(initializer, tf.Tensor):
                if util.shape(x=initializer) != shape:
                    raise TensorforceError.mismatch(
                        name='Module.add_variable', value1='shape', value2='initializer'
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
            variable.is_saved = is_saved

            if is_global:
                # Register variable as global tensor
                scoped_name = util.join_scopes(*get_global_scope(), name)
                assert scoped_name not in self.root.global_tensors_spec
                self.root.global_tensors_spec[scoped_name] = dict(type=dtype, shape=shape)
                assert len(self.root.graph.get_collection(name=scoped_name)) == 0
                self.root.graph.add_to_collection(name=scoped_name, value=variable)

        # get/assign operation (delayed for timestep)
        util.identity_operation(x=variable, operation_name=(name + '-output'))
        if name != 'timesteps':
            module = self
            while not module.is_root:
                module = module.parent
            variable.assign(value=module.assignment_input[dtype], name=(name + '-assign'))

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
        if shape is not None and (
            not util.is_iterable(x=shape) or not all(isinstance(dims, int) for dims in shape)
        ):
            raise TensorforceError.type(name='placeholder', argument='shape', value=shape)
        elif shape is not None and not all(dims > 0 for dims in shape):
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
            elif not util.is_dtype(x=default, dtype=dtype):
                raise TensorforceError.unexpected()

        # Placeholder
        if shape is None:
            assert not batched
        elif batched:
            shape = (None,) + shape
        if default is None:
            dtype = util.tf_dtype(dtype=dtype)
            placeholder = tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
        else:
            # check dtype and shape !!!
            assert shape is not None
            placeholder = tf.compat.v1.placeholder_with_default(
                input=default, shape=shape, name=name
            )

        return placeholder

    def is_summary_logged(self, label):
        # Check whether any summaries are logged
        module = self
        while module.summary_labels is None:
            module = module.parent

        if module.summary_labels is None:
            return False

        # # Check whether not in while loop
        # if Module.while_counter > 0:
        #     return False
        # # Check whether not in nested condition
        # if Module.cond_counter > 1:
        #     return False

        # Check whether given label is logged
        if util.is_iterable(x=label):
            assert all(not x.endswith('-histogram') for x in label)
            if module.summary_labels != 'all' and \
                    all(x not in module.summary_labels for x in label):
                return False
        else:
            if (module.summary_labels != 'all' or label.endswith('-histogram')) and \
                    label not in module.summary_labels:
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
                raise TensorforceError.value(
                    name='Module.add_summary', argument='label', value=label
                )
        else:
            if not isinstance(label, str):
                raise TensorforceError.type(
                    name='Module.add_summary', argument='label', dtype=type(label)
                )
        # name
        if not isinstance(name, str):
            raise TensorforceError.type(
                name='Module.add_summary', argument='name', dtype=type(name)
            )
        # tensor
        if not isinstance(tensor, (tf.Tensor, tf.Variable)):
            raise TensorforceError.type(
                name='Module.add_summary', argument='tensor', dtype=type(tensor)
            )
        # pass_tensors
        if util.is_iterable(x=pass_tensors):
            if not all(isinstance(x, (tf.Tensor, tf.IndexedSlices)) for x in pass_tensors):
                raise TensorforceError.value(
                    name='Module.add_summary', argument='pass_tensors', value=pass_tensors
                )
        elif pass_tensors is not None:
            if not isinstance(pass_tensors, tf.Tensor):
                raise TensorforceError.type(
                    name='Module.add_summary', argument='pass_tensors', dtype=type(pass_tensors)
                )
        # enumerate_last_rank
        if not isinstance(enumerate_last_rank, bool):
            raise TensorforceError.type(
                name='Module.add_summary', argument='enumerate_last_rank', dtype=type(tensor)
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

        # # Scope handling
        # if Module.scope_stack is not None:
        #     for scope in reversed(Module.scope_stack[1:]):
        #         scope.__exit__(None, None, None)
        #     if len(Module.global_scope) > 0:
        #         temp_scope = tf.name_scope(name='/'.join(Module.global_scope))
        #         temp_scope.__enter__()
        #     tensors = util.fmap(function=util.identity_operation, xs=tensors)

        # TensorFlow summaries
        assert Module.global_summary_step is not None
        step = self.root.timesteps  # self.global_tensor(name=Module.global_summary_step)
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

        # # Scope handling
        # if Module.scope_stack is not None:
        #     if len(Module.global_scope) > 0:
        #         temp_scope.__exit__(None, None, None)
        #     for scope in Module.scope_stack[1:]:
        #         scope.__enter__()

        with tf.control_dependencies(control_inputs=summaries):
            return util.fmap(function=util.identity_operation, xs=pass_tensors)

    @staticmethod
    def get_module_class_and_args(
        name, module=None, modules=None, default_module=None, disable_first_arg=False, **kwargs
    ):
        # name
        if not util.is_valid_name(name=name):
            raise TensorforceError.value(name='Module.add_module', argument='name', value=name)

        # modules
        if modules is not None and not isinstance(modules, dict):
            raise TensorforceError.type(
                name='Module.add_module', argument='modules', dtype=type(modules)
            )

        # default_module
        if default_module is not None and default_module not in modules and \
                not issubclass(default_module, Module):
            raise TensorforceError.value(
                name='Module.add_module', argument='default_module', value=default_module
            )

        # disable_first_arg
        if not isinstance(disable_first_arg, bool):
            raise TensorforceError.type(
                name='Module.add_module', argument='disable_first_arg',
                dtype=type(disable_first_arg)
            )

        # module
        if isinstance(module, dict):
            # Dictionary module specification (type either given via 'type' or 'default_module')
            util.deep_disjoint_update(target=kwargs, source=module)
            module = kwargs.pop('type', default_module)
            return Module.get_module_class_and_args(
                name=name, module=module, modules=modules, default_module=default_module,
                disable_first_arg=True, **kwargs
            )

        elif isinstance(module, str):
            if os.path.isfile(module):
                # JSON file module specification
                with open(module, 'r') as fp:
                    module = json.load(fp=fp)
                return Module.get_module_class_and_args(
                    name=name, module=module, modules=modules, default_module=default_module,
                    disable_first_arg=True, **kwargs
                )

            elif '.' in module:
                # Library module specification
                library_name, module_name = module.rsplit('.', 1)
                library = importlib.import_module(name=library_name)
                module = getattr(library, module_name)
                return Module.get_module_class_and_args(
                    name=name, module=module, modules=modules, default_module=default_module,
                    disable_first_arg=True, **kwargs
                )

            elif modules is not None and module in modules:
                # Keyword module specification
                return Module.get_module_class_and_args(
                    name=name, module=modules[module], modules=modules,
                    default_module=default_module, disable_first_arg=True, **kwargs
                )

            elif 'default' in modules or default_module is not None:
                # Default module specification
                if '_first_arg' in kwargs:
                    raise TensorforceError.invalid(name='Module.add_module', argument='_first_arg')
                if module is not None:
                    if disable_first_arg:
                        raise TensorforceError.value(
                            name='Module.add_module', argument='module', value=module
                        )
                    kwargs['_first_arg'] = module
                if default_module is None:
                    default_module = modules['default']
                return Module.get_module_class_and_args(
                    name=name, module=default_module, modules=modules, **kwargs
                )

            else:
                raise TensorforceError.value(
                    name='Module.add_module', argument='module', value=module
                )

        elif not callable(module) and ('default' in modules or default_module is not None):
            # Default module specification
            if '_first_arg' in kwargs:
                raise TensorforceError.invalid(name='Module.add_module', argument='_first_arg')
            if module is not None:
                kwargs['_first_arg'] = module
            if default_module is None:
                default_module = modules['default']
            return Module.get_module_class_and_args(
                name=name, module=default_module, modules=modules, **kwargs
            )

        elif callable(module):
            if '_first_arg' in kwargs:
                args = (kwargs.pop('_first_arg'),)
            else:
                args = ()
            kwargs['name'] = name
            return module, args, kwargs

        else:
            raise TensorforceError.value(name='Module.add_module', argument='module', value=module)

    def add_module(
        self, name, module=None, modules=None, default_module=None, is_trainable=True,
        is_saved=True, **kwargs
    ):
        # name
        if any(name == module.name for module in self.this_submodules):
            raise TensorforceError.exists(name='module', value=name)

        # is_trainable
        if not isinstance(is_trainable, bool):
            raise TensorforceError.type(
                name='Module.add_module', argument='is_trainable', dtype=type(is_trainable)
            )

        # is_saved
        if not isinstance(is_saved, bool):
            raise TensorforceError.type(
                name='Module.add_module', argument='is_saved', dtype=type(is_saved)
            )

        # module, modules, default_module
        module_cls, args, kwargs = Module.get_module_class_and_args(
            name=name, module=module, modules=modules, default_module=default_module, **kwargs
        )

        # Module constructor
        module = module_cls(*args, **kwargs)
        popped = MODULE_STACK.pop()
        assert popped is module

        assert not module.is_root
        assert not module.is_initialized
        if module.is_trainable is None:
            module.is_trainable = is_trainable
        else:
            assert is_trainable  # default
        if module.is_saved is None:
            module.is_saved = is_saved
        else:
            assert is_saved  # default

        return module

    def get_available_summaries(self):
        summaries = set(self.available_summaries)
        for module in self.modules.values():
            summaries.update(module.get_available_summaries())
        return sorted(summaries)

    @property
    def this_submodules(self):
        return list(self._flatten(recursive=False, predicate=(lambda x: isinstance(x, tf.Module))))

    @property
    def this_trainable_variables(self):
        return list(self._flatten(recursive=False, predicate=(
            lambda x: isinstance(x, tf.Variable) and getattr(x, 'trainable', False)
        )))

    @property
    def trainable_variables(self):
        variables = list(self._flatten(recursive=False, predicate=(
            lambda x: isinstance(x, tf.Variable) and getattr(x, 'trainable', False)
        )))
        for module in self.this_submodules:
            if getattr(module, 'is_trainable', True):
                variables.extend(module.trainable_variables)
        return variables

    @property
    def saved_variables(self):
        variables = list(self._flatten(recursive=False, predicate=(
            lambda x: isinstance(x, tf.Variable) and getattr(x, 'is_saved', True)
        )))
        for module in self.this_submodules:
            if not hasattr(module, 'is_saved'):
                variables.extend(module.variables)
            elif module.is_saved:
                variables.extend(module.saved_variables)
        return variables
