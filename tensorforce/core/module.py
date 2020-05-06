# Copyright 2020 Tensorforce Team. All Rights Reserved.
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
import os
import time

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
import tensorforce.core
from tensorforce.core import SignatureDict, TensorDict, tf_util


MODULE_STACK = list()


def get_global_scope():
    for n, module in enumerate(MODULE_STACK):
        assert isinstance(module, (Module, tf.name_scope))
        yield module.name
        # elif n == 1 and isinstance(module, str):
        #     yield module
        # else:
        #     raise TensorforceError.unexpected()


def make_key(*, x):
    try:
        hash(x)
        x < x
        return x
    except TypeError as exc:
        if isinstance(x, tuple) and len(x) > 0 and all(isinstance(y, tf.Variable) for y in x):
            return tuple(y.name for y in x)
        elif isinstance(x, Module):
            return x.__class__.__name__
        elif hasattr(x, '__name__'):
            return x.__name__
        else:
            raise exc


def tf_function(*, num_args):

    def decorator(function):

        def decorated(self, *args, **kwargs):
            assert len(args) == 0 or len(kwargs) == 0
            assert len(kwargs) >= num_args or len(args) == num_args

            # Function name and qualname
            name = function.__name__
            qualname = function.__qualname__
            assert qualname.endswith('.' + name)

            # Parameters-to-graph mapping
            if not hasattr(self, name + '_graphs'):
                setattr(self, name + '_graphs', OrderedDict())
            function_graphs = getattr(self, name + '_graphs')

            # Graph parameters
            graph_params = tuple(make_key(x=arg) for arg in list(kwargs.values())[num_args:])

            # Apply raw function if qualname mismatch, which indicates super() call
            # Call early to avoid check for number of arguments in case it has changed
            if graph_params in function_graphs and qualname != function_graphs[graph_params][0]:
                return function(self, *args, **kwargs)

            # Graph signature
            graph_signature = self.input_signature(function=name)
            assert graph_signature.num_args() == num_args

            # Graph arguments
            if len(kwargs) > 0:
                graph_args = graph_signature.kwargs_to_args(kwargs=kwargs)
            else:
                graph_args = args

            if graph_params not in function_graphs:
                # Check that length of graph specs are consistent
                assert len(function_graphs) == 0 or \
                    len(next(iter(function_graphs))) == len(graph_params)

                # Params kwargs
                params_kwargs = dict(list(kwargs.items())[num_args:])

                # Function graph
                def function_graph(*args):
                    if self not in MODULE_STACK:
                        MODULE_STACK.append(self)
                        pop_module_stack = True
                    else:
                        pop_module_stack = False
                    if self.device is not None:
                        self.device.__enter__()
                    results = Module.with_name_scope(method=function)(
                        self, **graph_signature.args_to_kwargs(args=args).to_kwargs(),
                        **params_kwargs
                    )
                    if self.device is not None:
                        self.device.__exit__(None, None, None)
                    if pop_module_stack:
                        popped = MODULE_STACK.pop()
                        assert popped is self
                    return results

                function_graphs[graph_params] = (qualname, tf.function(
                    func=function_graph, input_signature=graph_signature.to_list(), autograph=False
                    # experimental_implements=None, experimental_autograph_options=None,
                    # experimental_relax_shapes=False, experimental_compile=None
                ))

            # Apply function graph
            return function_graphs[graph_params][1](*graph_args)

        # TensorFlow make_decorator
        return tf.compat.v1.flags.tf_decorator.make_decorator(
            target=function, decorator_func=decorated
        )

    return decorator


class Module(tf.Module):
    """
    Base class for modules.

    Args:
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    _TF_MODULE_IGNORED_PROPERTIES = frozenset((
        '_self_unconditional_checkpoint_dependencies',
        '_self_unconditional_dependency_names',
        'parent'
    ))

    global_summary_step = None

    def __init__(
        self, *, is_root=False, device=None, summary_labels=None, l2_regularization=None, name=None
    ):
        super().__init__(name=name)

        self.is_root = is_root

        if self.is_root:
            MODULE_STACK.clear()
            MODULE_STACK.append(self)
            if device is None:
                self.device = None
            else:
                self.device = tf.device(device_name=device)
            self.is_initialized = None
            self.is_trainable = True
            self.is_saved = True
            self.global_tensors_spec = OrderedDict()
            self.input_tensors = None
            self.output_tensors = None
            self.query_tensors = None
            self.available_summaries = None
        else:
            assert len(MODULE_STACK) > 1
            assert not isinstance(MODULE_STACK[-1], type) or isinstance(self, MODULE_STACK[-1])
            self.parent = MODULE_STACK[-2]
            MODULE_STACK[-1] = self
            if device is None:
                self.device = self.parent.device
            else:
                self.device = tf.device(device_name=device)
            self.is_trainable = None
            self.is_saved = None

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

    def initialize(self):
        for module in self.this_submodules:
            if isinstance(module, Module):
                MODULE_STACK.append(module)
                with tf.name_scope(name=module.name):
                    module.initialize()
                popped = MODULE_STACK.pop()
                assert popped is module

    def root_initialize(self):
        # Check whether module is root and not already initialized
        assert self.is_root and not self.is_initialized
        assert len(MODULE_STACK) == 1 and MODULE_STACK[0] is self

        # Set internal attributes
        self.input_tensors = OrderedDict()
        self.output_tensors = OrderedDict()
        self.query_tensors = OrderedDict()
        self.available_summaries = set()

        # TensorFlow device
        if self.device is not None:
            self.device.__enter__()

        with self.name_scope:

            if self.is_root:
                # Timestep counter  (TODO: why earlier? still required?)
                self.timesteps = self.variable(
                    name='timesteps', dtype='int', shape=(), initializer='zeros',
                    is_trainable=False, is_saved=True
                )

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
                    condition = tf_util.constant(value=True, dtype='bool')
                    record_summaries = tf.summary.record_if(condition=condition)
                    record_summaries.__enter__()

                # # Assignment values
                # self.assignment_input = dict(
                #     bool=self.add_placeholder(
                #         name='assignment-bool', dtype='bool', shape=None, batched=False
                #     ), int=self.add_placeholder(
                #         name='assignment-int', dtype='int', shape=None, batched=False
                #     ), long=self.add_placeholder(
                #         name='assignment-long', dtype='int', shape=None, batched=False
                #     ), float=self.add_placeholder(
                #         name='assignment-float', dtype='float', shape=None, batched=False
                #     )
                # )

                # Episode counter
                self.episodes = self.variable(
                    name='episodes', dtype='int', shape=(), initializer='zeros',
                    is_trainable=False, is_saved=True
                )

                # Update counter
                self.updates = self.variable(
                    name='updates', dtype='int', shape=(), initializer='zeros', is_trainable=False,
                    is_saved=True
                )

                if self.summarizer_spec is not None:
                    if len(self.summarizer_spec.get('custom', ())) > 0:
                        self.summarize_input = self.add_placeholder(
                            name='summarize', dtype='float', shape=None, batched=False
                        )
                        # self.summarize_step_input = self.add_placeholder(
                        #     name='summarize-step', dtype='int', shape=(), batched=False,
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
                        condition = tf_util.constant(value=True, dtype='bool')

                    elif 'variables' in self.summarizer_spec['frequency']:
                        step = self.global_tensor(name=Module.global_summary_step)
                        frequency = tf_util.constant(
                            value=self.summarizer_spec['frequency']['variables'], dtype='int'
                        )
                        zero = tf_util.constant(value=0, dtype='int')
                        condition = (
                            lambda: tf.math.equal(x=tf.math.mod(x=step, y=frequency), y=zero)
                        )

                    else:
                        condition = tf_util.constant(value=False, dtype='bool')

                    record_summaries = tf.summary.record_if(condition=condition)
                    record_summaries.__enter__()

            self.initialize()

        if self.is_root and self.summarizer_spec is not None:
            record_summaries.__exit__(None, None, None)
            Module.global_summary_step = None

        # possibility to turn off?
        if self.saver_spec is None:
            max_to_keep = 5
        else:
            max_to_keep = self.saver_spec.get('max-checkpoints', 5)
        self.saver = tf.compat.v1.train.Saver(
            var_list=self.saved_variables,  # should be given?
            reshape=False,
            sharded=False,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            saver_def=None,
            builder=None,
            defer_build=False,
            allow_empty=False,
            write_version=tf.compat.v1.train.SaverDef.V2,
            pad_step_number=False,
            save_relative_paths=False,
            filename=None
        )

        # # global_variables += [self.global_episode, self.global_timestep]
        # init_op = tf.compat.v1.variables_initializer(var_list=tf.compat.v1.global_variables())
        # if self.summarizer_spec is not None:
        #     init_op = tf.group(init_op, self.summarizer_init)
        # if self.graph_summary is None:
        #     ready_op = tf.compat.v1.report_uninitialized_variables(var_list=self.variables)
        #     ready_for_local_init_op = None
        #     local_init_op = None
        # else:
        #     ready_op = None
        #     ready_for_local_init_op = tf.compat.v1.report_uninitialized_variables(
        #         var_list=self.variables
        #     )
        #     local_init_op = self.graph_summary

        # def init_fn(scaffold, session):
        #     if self.saver_spec is not None and self.saver_spec.get('load', True):
        #         directory = self.saver_spec['directory']
        #         load = self.saver_spec.get('load')
        #         if isinstance(load, str):
        #             save_path = os.path.join(directory, load)
        #         else:
        #             save_path = tf.compat.v1.train.latest_checkpoint(
        #                 checkpoint_dir=directory, latest_filename=None
        #             )
        #         if save_path is not None:
        #             # global vs local model restored correctly?
        #             scaffold.saver.restore(sess=session, save_path=save_path)
        #             session.run(fetches=util.join_scopes(self.name + '.reset', 'timestep-output:0'))

        # TensorFlow scaffold object
        # TODO explain what it does.
        # self.scaffold = tf.compat.v1.train.Scaffold(
        #     init_op=init_op,
        #     init_feed_dict=None,
        #     init_fn=init_fn,
        #     ready_op=ready_op,
        #     ready_for_local_init_op=ready_for_local_init_op,
        #     local_init_op=local_init_op,
        #     summary_op=None,
        #     saver=self.saver,
        #     copy_from_scaffold=None
        # )

        # Checkpoint saver hook
        if self.saver_spec is not None:
            self.saver_directory = self.saver_spec['directory']
            self.saver_filename = self.saver_spec.get('filename', self.name)
            frequency = self.saver_spec.get('frequency', 600)
            if frequency is not None:
                hooks = [tf.compat.v1.train.CheckpointSaverHook(
                    checkpoint_dir=self.saver_directory, save_secs=frequency, save_steps=None,
                    saver=None,  # None since given via 'scaffold' argument.
                    checkpoint_basename=self.saver_filename, scaffold=self.scaffold, listeners=None
                )]
        else:
            hooks = list()
            self.saver_directory = None
            self.saver_filename = self.name

        # TensorFlow non-distributed monitored session object
        # self.monitored_session = tf.compat.v1.train.SingularMonitoredSession(
        #     hooks=hooks,
        #     scaffold=self.scaffold,
        #     master='',  # Default value.
        #     config=self.execution.get('session_config'),
        #     checkpoint_dir=None
        # )

        # graph_default_context.__exit__(None, None, None)
        # self.graph.finalize()

        # self.monitored_session.__enter__()
        # self.session = self.monitored_session._tf_sess()

        if self.saver_directory is not None:
            self.save(
                directory=self.saver_directory, filename=self.saver_filename, format='tensorflow',
                append='timesteps', no_act_pb=True
            )

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
                        condition = tf_util.constant(value=True, dtype='bool')

                    elif isinstance(self.summarizer_spec['frequency'], int):
                        if function_name in ('act', 'independent_act'):
                            step = self.timesteps
                            frequency = tf_util.constant(
                                value=self.summarizer_spec['frequency'], dtype='int'
                            )
                            zero = tf_util.constant(value=0, dtype='int')
                            condition = (
                                lambda: tf.math.equal(x=tf.math.mod(x=step, y=frequency), y=zero)
                            )
                        elif function_name in ('reset', 'independent_act'):
                            condition = tf_util.constant(value=False, dtype='bool')
                        else:
                            condition = tf_util.constant(value=True, dtype='bool')

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
                        frequency = tf_util.constant(
                            value=self.summarizer_spec['frequency'][function_name], dtype='int'
                        )
                        zero = tf_util.constant(value=0, dtype='int')
                        condition = (
                            lambda: tf.math.equal(x=tf.math.mod(x=step, y=frequency), y=zero)
                        )

                    else:
                        condition = tf_util.constant(value=False, dtype='bool')

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

    def input_signature(self, *, function):
        if function == 'regularize':
            return SignatureDict()

        else:
            raise NotImplementedError

    @tf_function(num_args=0)
    def regularize(self):
        zero = tf_util.constant(value=0.0, dtype='float')

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
                    variable = tf_util.cast(x=variable, dtype='float')
                    l2_variables.append(tf.reduce_sum(input_tensor=tf.square(x=variable)))
                return l2_regularization * tf.math.add_n(inputs=l2_variables)

            skip_l2_regularization = tf.math.equal(x=l2_regularization, y=zero)
            regularization_loss = tf.cond(
                pred=skip_l2_regularization, true_fn=no_l2_regularization,
                false_fn=apply_l2_regularization
            )

        for module in self.this_submodules:
            if isinstance(module, Module) and module.is_trainable:
                regularization_loss += module.regularize()

        return regularization_loss

    def create_api_function(self, *, name, api_function):
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

        # # Function-level identity operation for retrieval
        # query_tensors = set()
        # # for scoped_name, tensor in Module.queryable_tensors.items():
        # #     tensor = util.identity_operation(x=tensor, operation_name=(scoped_name + '-query'))
        # #     assert tensor.name.endswith('-query:0')
        # #     assert scoped_name not in query_tensors
        # #     query_tensors.add(scoped_name)
        # for scoped_name in self.global_tensors_spec:
        #     scoped_name1 = scoped_name.replace('agent/', '')
        #     scoped_name2 = scoped_name.replace('agent/', name.replace('.', '/') + '/')
        #     collection = self.root.graph.get_collection(name=scoped_name2)
        #     if len(collection) == 0:
        #         continue
        #     tensor = util.identity_operation(x=collection[0], operation_name=(scoped_name1 + '-query'))
        #     assert tensor.name.endswith('-query:0')
        #     assert scoped_name not in query_tensors
        #     query_tensors.add(scoped_name)
        # self.query_tensors[name[name.index('.') + 1:]] = sorted(query_tensors)

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

    def set_global_tensor(self, *, name, tensor):
        assert self.root.is_initialized

        if not isinstance(tensor, tf.Tensor):
            raise TensorforceError.unexpected()

        spec = dict(type=tf_util.dtype(x=tensor), shape=tf_util.shape(x=tensor))
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

    def global_tensor(self, *, name):
        assert self.root.is_initialized

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

    def variable(self, *, name, dtype, shape, initializer, is_trainable, is_saved):
        assert not self.root.is_initialized
        # name
        if not isinstance(name, str):
            raise TensorforceError.type(name='variable', argument='name', dtype=type(name))
        # dtype
        if dtype not in ('bool', 'int', 'float'):
            raise TensorforceError.value(name='variable', argument='dtype', value=dtype)
        # shape
        if not util.is_iterable(x=shape) or not all(isinstance(dims, int) for dims in shape):
            raise TensorforceError.value(name='variable', argument='shape', value=shape)
        elif not all(dims > 0 for dims in shape):
            raise TensorforceError.value(name='variable', argument='shape', value=shape)
        # is_trainable
        if not isinstance(is_trainable, bool):
            raise TensorforceError.type(
                name='variable', argument='is_trainable', dtype=type(is_trainable)
            )
        elif is_trainable and dtype != 'float':
            raise TensorforceError.value(
                name='variable', argument='is_trainable', value=is_trainable,
                condition='dtype != float'
            )
        # initializer
        initializer_names = (
            'normal', 'normal-relu', 'orthogonal', 'orthogonal-relu', 'zeros', 'ones'
        )
        if not isinstance(initializer, (util.py_dtype(dtype=dtype), np.ndarray, tf.Tensor)) and \
                initializer not in initializer_names:
            raise TensorforceError.value(name='variable', argument='initializer', value=initializer)
        elif isinstance(initializer, np.ndarray) and \
                initializer.dtype != util.np_dtype(dtype=dtype):
            raise TensorforceError.type(
                name='variable', argument='initializer', dtype=type(initializer)
            )
        elif isinstance(initializer, tf.Tensor) and tf_util.dtype(x=initializer) != dtype:
            raise TensorforceError.type(
                name='variable', argument='initializer', dtype=type(initializer)
            )
        # is_saved
        if not isinstance(is_saved, bool):
            raise TensorforceError.type(name='variable', argument='is_saved', dtype=type(is_saved))
        # Variable initializer
        if self.device is not None:
            self.device.__enter__()
        if isinstance(initializer, util.py_dtype(dtype=dtype)):
            initializer = tf_util.constant(value=initializer, dtype=dtype, shape=shape)
        elif isinstance(initializer, np.ndarray):
            if initializer.shape != shape:
                raise TensorforceError.mismatch(
                    name='Module.variable', value1='shape', value2='initializer'
                )
            initializer = tf_util.constant(value=initializer, dtype=dtype)
        elif isinstance(initializer, tf.Tensor):
            if tf_util.shape(x=initializer) != shape:
                raise TensorforceError.mismatch(
                    name='Module.variable', value1='shape', value2='initializer'
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
                stddev = min(0.1, np.sqrt(2.0 / util.product(xs=shape[:-1])))
            else:
                stddev = min(0.1, np.sqrt(2.0 / (util.product(xs=shape[:-1]) + shape[-1])))
            initializer = tf.random.normal(
                shape=shape, stddev=stddev, dtype=tf_util.get_dtype(type=dtype)
            )
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
                orthogonal = orthogonal * np.sqrt(2.0)
            initializer = tf_util.constant(value=orthogonal.reshape(shape), dtype=dtype)
        elif initializer == 'zeros':
            initializer = tf_util.zeros(shape=shape, dtype=dtype)
        elif initializer == 'ones':
            initializer = tf_util.ones(shape=shape, dtype=dtype)

        # Variable
        variable = tf.Variable(
            initial_value=initializer, trainable=is_trainable, validate_shape=True, name=name,
            dtype=tf_util.get_dtype(type=dtype), shape=shape
        )
        variable.is_saved = is_saved

        if self.device is not None:
            self.device.__exit__(None, None, None)

        # if is_global:
        #     # Register variable as global tensor
        #     scoped_name = util.join_scopes(*get_global_scope(), name)
        #     assert scoped_name not in self.root.global_tensors_spec
        #     self.root.global_tensors_spec[scoped_name] = dict(type=dtype, shape=shape)
        #     # assert len(self.root.graph.get_collection(name=scoped_name)) == 0
        #     # self.root.graph.add_to_collection(name=scoped_name, value=variable)

        # get/assign operation (delayed for timestep)
        # util.identity_operation(x=variable, operation_name=(name + '-output'))
        # if name != 'timesteps':
        #     module = self
        #     while not module.is_root:
        #         module = module.parent
        #     variable.assign(value=module.assignment_input[dtype], name=(name + '-assign'))

        # Add summary
        if is_trainable:
            variable = self.add_summary(
                label='variables', name=name, tensor=variable, mean_variance=True
            )
            variable = self.add_summary(label='variables-histogram', name=name, tensor=variable)

        return variable

    def is_summary_logged(self, *, label):
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
        self, *, label, name, tensor, pass_tensors=None, return_summaries=False,
        mean_variance=False, enumerate_last_rank=False
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
            dims = tf_util.shape(x=tensor)[-1]
            tensors = TensorDict(((name + str(n), tensor[..., n]) for n in range(dims)))
        else:
            tensors = TensorDict([(name, tensor)])

        if mean_variance:
            for name in list(tensors):
                tensor = tensors.pop(name)
                mean, variance = tf.nn.moments(x=tensor, axes=tuple(range(tf_util.rank(x=tensor))))
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
            shape = tf_util.shape(x=tensor)
            if shape == ():
                summaries.append(tf.summary.scalar(name=name, data=tensor, step=step))
            elif shape == (-1,):
                tensor = tf.math.reduce_sum(input_tensor=tensor, axis=0)
                summaries.append(tf.summary.scalar(name=name, data=tensor, step=step))
            elif shape == (1,):
                tensor = tf.squeeze(input=tensor, axis=0)
                summaries.append(tf.summary.scalar(name=name, data=tensor, step=step))
            elif shape == (-1, 1):
                tensor = tf.math.reduce_sum(input_tensor=tf.squeeze(input=tensor, axis=1), axis=0)
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
            return util.fmap(function=tf_util.identity, xs=pass_tensors)

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
            if not isinstance(module, Module) or module.is_trainable:
                variables.extend(module.trainable_variables)
        return variables

    @property
    def saved_variables(self):
        variables = list(self._flatten(recursive=False, predicate=(
            lambda x: isinstance(x, tf.Variable) and getattr(x, 'is_saved', True)
        )))
        for module in self.this_submodules:
            if not isinstance(module, Module):
                variables.extend(module.variables)
            elif module.is_saved:
                variables.extend(module.saved_variables)
        return variables



    @staticmethod
    def get_module_class_and_args(
        *, name, module=None, modules=None, default_module=None, disable_first_arg=False, **kwargs
    ):
        # name
        if not isinstance(name, str):
            raise TensorforceError.type(name='Module.add_module', argument='name', dtype=type(name))
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
        self, *, name, module=None, modules=None, default_module=None, is_trainable=True,
        is_saved=True, **kwargs
    ):
        assert self.root.is_initialized is None

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
        MODULE_STACK.append(module_cls)
        module = module_cls(*args, **kwargs)
        popped = MODULE_STACK.pop()
        assert popped is module

        assert not module.is_root

        assert module.is_trainable is None
        module.is_trainable = is_trainable
        # else:
        #     assert is_trainable  # default
        assert module.is_saved is None
        module.is_saved = is_saved
        # else:
        #     assert is_saved  # default

        return module
