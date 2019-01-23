# Copyright 2018 TensorForce Team. All Rights Reserved.
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

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import while_v2

from tensorforce import TensorforceError, util


tf.enable_resource_variables()


class Module(object):
    """
    Base class for TensorFlow modules.
    """

    global_scope = None
    global_tensors_spec = None
    global_tensors = None  # per agent, main module, or so

    @staticmethod
    def register_tensor(name, spec, batched):
        if '/' in name:
            raise TensorforceError.value(name='name', value=name)

        if Module.global_scope is None:  # ???
            raise TensorforceError.unexpected()

        # scoped_name = util.join_scopes(*Module.global_scope, name)
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
    def update_tensor(name, tensor):
        # for n in range(len(Module.global_scope) + 1):
            # partial_scope = Module.global_scope[:len(Module.global_scope) - n]
            # scoped_name = util.join_scopes(*partial_scope, name)
        #     if scoped_name in Module.global_tensors_spec:
        #         break
        # else:
        #     raise TensorforceError("Global tensor is not registered: {}.".format(name))
        if name not in Module.global_tensors_spec:
            raise TensorforceError("Global tensor is not registered: {}.".format(name))

        scoped_name = name
        spec = Module.global_tensors_spec[scoped_name]

        if not util.is_consistent_with_value_spec(value_spec=spec, x=tensor):
            raise TensorforceError("Invalid overwriting tensor: {}.".format(tensor))

        scoped_name = util.join_scopes(*Module.global_scope, name)

        previous = Module.global_tensors.get(scoped_name)
        Module.global_tensors[scoped_name] = tensor

        return previous

    @staticmethod
    def update_tensors(**kwargs):
        for name, tensor in kwargs.items():
            Module.update_tensor(name=name, tensor=tensor)

    @staticmethod
    def retrieve_tensor(name):
        # for n in range(len(Module.global_scope) + 1):
            # partial_scope = Module.global_scope[:len(Module.global_scope) - n]
            # scoped_name = util.join_scopes(*partial_scope, name)
        #     if scoped_name in Module.global_tensors_spec:
        #         break
        # else:
        #     raise TensorforceError("Global tensor is not registered: {}.".format(name))
        if name not in Module.global_tensors_spec:
            raise TensorforceError("Global tensor is not registered: {}.".format(name))

        for n in range(len(Module.global_scope) + 1):
            partial_scope = Module.global_scope[:len(Module.global_scope) - n]
            scoped_name = util.join_scopes(*partial_scope, name)
            if scoped_name in Module.global_tensors:
                break
        else:
            raise TensorforceError("Global tensor is not set: {}.".format(name))

        # scoped_name = util.join_scopes(*Module.global_scope, name)

        # if scoped_name not in Module.global_tensors:
        #     raise TensorforceError("Global tensor is not set: {}.".format(scoped_name))

        return Module.global_tensors[scoped_name]

    is_add_module = False

    # Inherit arguments
    inherit_l2_regularization = None
    inherit_summary_labels = None

    def __init__(self, name, l2_regularization=None, summary_labels=None, device=None):
        # name
        if not util.is_valid_name(name=name):
            raise TensorforceError.value(name='module', argument='name', value=name)
        # l2_regularization
        if l2_regularization is not None and not isinstance(l2_regularization, float):
            raise TensorforceError.type(
                name='module', argument='l2_regularization', value=l2_regularization
            )
        if l2_regularization is not None and l2_regularization < 0.0:
            raise TensorforceError.value(
                name='module', argument='l2_regularization', value=l2_regularization
            )
        # summary_labels
        if summary_labels is not None and \
                not all(isinstance(label, str) for label in summary_labels):
            raise TensorforceError.type(
                name='module', argument='summary_labels', value=summary_labels
            )
        # device
        # ???

        # Attributes specified via constructor arguments
        self.name = name
        self.l2_regularization = l2_regularization
        self.summary_labels = None if summary_labels is None else set(summary_labels)
        self.device = device

        # Otherwise inherit arguments
        if self.l2_regularization is None:
            self.l2_regularization = Module.inherit_l2_regularization
        if self.summary_labels is None:
            self.summary_labels = Module.inherit_summary_labels

        # Internal attributes
        self.parent = None
        self.scope = None
        self.is_subscope = None
        self.modules = OrderedDict()
        self.trainable_modules = OrderedDict()
        self.is_initialized = False
        self.variables = None
        self.trainable_variables = None
        self.regularized_variables = None

        if not Module.is_add_module:
            Module.global_scope = list()
            Module.global_tensors_spec = OrderedDict()

    def tf_initialize(self):
        pass

    def tf_regularize(self):
        regularization_loss = tf.zeros(shape=(), dtype=util.tf_dtype(dtype='float'))

        if self.l2_regularization is not None and self.l2_regularization > 0.0:
            l2_variables = [
                tf.reduce_sum(input_tensor=tf.square(x=variable))
                for variable in self.trainable_variables.values()
            ]
            regularization_loss += self.l2_regularization * tf.math.accumulate_n(
                inputs=l2_variables, shape=(), tensor_dtype=util.tf_dtype(dtype='float')
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
        self.regularized_variables = OrderedDict()

        if self.parent is None:
            Module.global_scope = list()

        Module.global_scope.append(self.name)

        # TensorFlow device and variable scope
        self.scope = tf.variable_scope(name_or_scope=self.name, use_resource=True)
        if self.device is not None:
            self.device = tf.device(device_name_or_function=self.device)
            self.device.__enter__()

        with self.scope:
            if self.parent is None:
                # Global timestep
                self.global_timestep = self.add_variable(
                    name='global-timestep', dtype='long', shape=(), is_trainable=False,
                    initializer='zeros', shared='global-timestep'
                )
                collection = tf.get_collection(key=tf.GraphKeys.GLOBAL_STEP)
                if len(collection) == 0:
                    tf.add_to_collection(
                        name=tf.GraphKeys.GLOBAL_STEP, value=self.global_timestep
                    )

                # Global episode
                self.global_episode = self.add_variable(
                    name='global-episode', dtype='long', shape=(), is_trainable=False,
                    initializer='zeros', shared='global-episode'
                )

                Module.global_tensors = OrderedDict(
                    timestep=self.global_timestep, episode=self.global_episode
                )

            for module in self.modules.values():
                module.initialize()
            self.tf_initialize()

        if self.device is not None:
            self.device.__exit__(None, None, None)

        Module.global_scope.pop()

        if self.parent is None:
            assert len(Module.global_scope) == 0
            Module.global_tensors = None
            Module.global_scope = None

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

    def create_tf_function(self, name, tf_function):
        # Call internal TensorFlow function
        def fn(*args, **kwargs):
            if self.is_subscope:
                Module.global_scope.append(self.name)
            if self.device is not None:
                self.device.__enter__()
            with tf.name_scope(name=name):
                results = tf_function(*args, **kwargs)
            if self.device is not None:
                self.device.__exit__(None, None, None)
            if self.is_subscope:
                Module.global_scope.pop()
            return results

        return fn

    def create_api_function(self, name, api_function):
        # Call API TensorFlow function
        Module.global_scope = list()
        Module.global_tensors = OrderedDict()
        if self.device is not None:
            self.device.__enter__()
        with tf.name_scope(name=name):
            results = api_function()

            # Function-level identity operation for retrieval
            for scoped_name, tensor in Module.global_tensors.items():
                if '/cond/' not in scoped_name and '/while/' not in scoped_name:
                    util.identity_operation(x=tensor, operation_name=(scoped_name + '-output'))

        if self.device is not None:
            self.device.__exit__(None, None, None)
        Module.global_tensors = None
        Module.global_scope = None

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
        Module.global_scope.append('cond')
        x = tf.cond(pred=pred, true_fn=true_fn, false_fn=false_fn, strict=True)
        Module.global_scope.pop()
        return x

    def while_loop(
        self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10,
        back_prop=False, swap_memory=False, maximum_iterations=None, return_same_structure=False,
        use_while_v2=False
    ):
        Module.global_scope.append('while')
        if use_while_v2:
            x = while_v2.while_loop(cond=cond, body=body, loop_vars=loop_vars)
        else:
            x = tf.while_loop(
                cond=cond, body=body, loop_vars=loop_vars, shape_invariants=shape_invariants,
                parallel_iterations=parallel_iterations, back_prop=back_prop,
                swap_memory=swap_memory, maximum_iterations=maximum_iterations,
                return_same_structure=return_same_structure
            )
        Module.global_scope.pop()
        return x

    def add_variable(
        self, name, dtype, shape, is_trainable, initializer='zeros', summarize=None, shared=None
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
        if not util.is_iterable(x=shape) or \
                not all(isinstance(num_dims, int) for num_dims in shape):
            raise TensorforceError.type(name='variable', argument='shape', value=shape)
        elif not all(num_dims > 0 for num_dims in shape):
            raise TensorforceError.value(name='variable', argument='shape', value=shape)
        # is_trainable
        if not isinstance(is_trainable, bool):
            raise TensorforceError.type(
                name='variable', argument='is_trainable', value=is_trainable
            )
        # initializer
        if not isinstance(initializer, (util.py_dtype(dtype=dtype), np.ndarray, tf.Tensor)) and \
                initializer not in ('random', 'zeros', 'ones'):
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
        elif isinstance(initializer, str) and initializer == 'random' and dtype != 'float':
            raise TensorforceError(
                message="Invalid variable initializer value for non-float variable: {}.".format(
                    initializer
                )
            )
        # summarize
        if summarize is not None and not isinstance(summarize, bool):
            raise TensorforceError.type(name='variable', argument='summarize', value=summarize)
        # shared
        if shared is not None and not isinstance(shared, str):
            raise TensorforceError.type(name='variable', argument='shared', value=shared)

        variable = None

        if shared is not None and len(tf.get_collection(key=shared)) > 0:
            # Retrieve shared variable from TensorFlow
            collection = tf.get_collection(key=shared)
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
                initializer = initializer
            elif isinstance(initializer, tf.Tensor):
                if util.shape(x=initializer) != shape:
                    raise TensorforceError(
                        "Invalid variable initializer shape: {}.".format(util.shape(x=initializer))
                    )
                initializer = initializer
            elif not isinstance(initializer, str):
                raise TensorforceError("Invalid variable initializer: {}".format(initializer))
            elif initializer == 'random':
                initializer = tf.random_normal(
                    shape=shape, mean=0.0, stddev=1e-2, dtype=util.tf_dtype(dtype=dtype)
                )
            elif initializer == 'zeros':
                initializer = tf.zeros(shape=shape, dtype=tf_dtype)
            elif initializer == 'ones':
                initializer = tf.ones(shape=shape, dtype=tf_dtype)

            # Variable
            variable = tf.Variable(
                initial_value=initializer, trainable=is_trainable, validate_shape=True, name=name,
                dtype=tf_dtype, expected_shape=shape, use_resource=True
            )  # collections=

            # Register shared variable with TensorFlow
            if shared is not None:
                tf.add_to_collection(name=shared, value=variable)

        # Register variable
        self.variables[name] = variable
        if is_trainable:
            self.trainable_variables[name] = variable

        # Add summary
        if (summarize is None and is_trainable) or summarize:
            variable = tf.identity(input=variable)
            variable = self.add_summary(
                label='variables', name=name, tensor=variable, mean_variance=True
            )

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
        if not util.is_iterable(x=shape) or \
                not all(isinstance(num_dims, int) for num_dims in shape):
            raise TensorforceError.type(name='placeholder', argument='shape', value=shape)
        elif not all(num_dims > 0 for num_dims in shape):
            raise TensorforceError.value(name='placeholder', argument='shape', value=shape)
        # batched
        if not isinstance(batched, bool):
            raise TensorforceError.type(name='placeholder', argument='batched', value=batched)
        # default
        if default is not None:
            if batched:
                raise TensorforceError.unexpected()
            elif not isinstance(default, tf.Tensor):
                raise TensorforceError.unexpected()
            elif util.dtype(x=default) != dtype:
                raise TensorforceError.unexpected()

        # Placeholder
        if batched:
            shape = (None,) + shape
        if default is None:
            dtype = util.tf_dtype(dtype=dtype)
            placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
        else:
            # check dtype and shape !!!
            placeholder = tf.placeholder_with_default(input=default, shape=shape, name=name)

        return placeholder

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
        if not isinstance(tensor, tf.Tensor):
            raise TensorforceError.type(name='summary', argument='tensor', value=tensor)
        # pass_tensors
        if util.is_iterable(x=pass_tensors):
            if not all(isinstance(x, tf.Tensor) for x in pass_tensors):
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

        # Check whether summaries are logged
        if self.summary_labels is None:
            return pass_tensors

        # Check whether not in while loop
        if 'while' in Module.global_scope:  # 'cond' in Module.global_scope
            return pass_tensors

        # Check whether given label is logged
        if util.is_iterable(x=label):
            if all(x not in self.summary_labels for x in label):
                return pass_tensors
        else:
            if label not in self.summary_labels:
                return pass_tensors

        # Handle enumerate_last_rank
        if enumerate_last_rank:
            num_dims = util.shape(x=tensor)[-1]
            tensors = OrderedDict([(name + str(n), tensor[..., n]) for n in range(num_dims)])
        else:
            tensors = OrderedDict([(name, tensor)])

        if mean_variance:
            for name in list(tensors):
                tensor = tensors.pop(name)
                mean, variance = tf.nn.moments(x=tensor, axes=tuple(range(util.rank(x=tensor))))
                tensors[name + '-mean'] = mean
                tensors[name + '-variance'] = variance

        # TensorFlow summaries
        summaries = list()
        for name, tensor in tensors.items():
            shape = util.shape(x=tensor)
            if shape == () or shape == (-1,):
                # Scalar
                summaries.append(tf.contrib.summary.scalar(name=name, tensor=tensor))
            elif shape == (1,) or shape == (-1, 1):
                # Single-value tensor as scalar
                tensor = tf.squeeze(input=tensor, axis=-1)
                summaries.append(tf.contrib.summary.scalar(name=name, tensor=tensor))
            else:
                # General tensor as histogram
                summaries.append(tf.contrib.summary.histogram(name=name, tensor=tensor))

        with tf.control_dependencies(control_inputs=summaries):
            if util.is_iterable(x=pass_tensors):
                return tuple(util.identity_operation(x=x) for x in pass_tensors)
            else:
                return util.identity_operation(x=pass_tensors)

    def add_module(
        self, name, module, modules=None, default_module=None, is_trainable=True,
        is_subscope=False, **kwargs
    ):
        # name
        if not util.is_valid_name(name=name):
            raise TensorforceError.value(name='module', argument='name', value=name)
        elif name in self.modules:
            raise TensorforceError.exists(name='sub-module', value=name)
        # module
        # ???
        # modules
        if modules is not None and not isinstance(modules, dict):
            raise TensorforceError.type(name='module', argument='modules', value=modules)
        # default_module
        # ???
        # is_trainable
        if not isinstance(is_trainable, bool):
            raise TensorforceError.type(name='module', argument='is_trainable', value=is_trainable)

        if isinstance(module, dict):
            # Dictionary module specification (type either given via 'type' or 'default_module')
            for key, value in module.items():
                if key in kwargs and kwargs[key] != value:
                    raise TensorforceError.mismatch(
                        name='module', argument=key, value1=kwargs[key], value2=value
                    )
                kwargs[key] = value
            module = kwargs.pop('type', default_module)
            return self.add_module(
                name=name, module=module, modules=modules, default_module=default_module,
                is_trainable=is_trainable, is_subscope=is_subscope, **kwargs
            )

        elif isinstance(module, str):
            if os.path.isfile(module):
                # JSON file module specification
                with open(module, 'r') as fp:
                    module = json.load(fp=fp)
                return self.add_module(
                    name=name, module=module, modules=modules, default_module=default_module,
                    is_trainable=is_trainable, is_subscope=is_subscope, **kwargs
                )

            elif '.' in module:
                # Library module specification
                library_name, module_name = module.rsplit('.', 1)
                library = importlib.import_module(name=library_name)
                module = getattr(library, module_name)
                return self.add_module(
                    name=name, module=module, modules=modules, default_module=default_module,
                    is_trainable=is_trainable, is_subscope=is_subscope, **kwargs
                )

            elif modules is not None and module in modules:
                # Keyword module specification
                return self.add_module(
                    name=name, module=modules[module], default_module=default_module,
                    is_trainable=is_trainable, is_subscope=is_subscope, **kwargs
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
            return self.add_module(
                name=name, module=default_module, modules=modules, is_trainable=is_trainable,
                is_subscope=is_subscope, **kwargs
            )

        elif callable(module):
            # Final callable module specification
            if Module.global_scope is None:
                raise TensorforceError.unexpected()

            # Global scope handling
            Module.is_add_module = True
            if is_subscope:
                Module.global_scope.append(name)

            # Inherit arguments
            Module.inherit_l2_regularization = self.l2_regularization
            Module.inherit_summary_labels = self.summary_labels

            # Module constructor
            if '_first_arg' in kwargs:
                module = module(name, kwargs.pop('_first_arg'), **kwargs)
            else:
                module = module(name=name, **kwargs)

            # Inherit arguments
            Module.inherit_l2_regularization = None
            Module.inherit_summary_labels = None

            # Global scope handling
            if is_subscope:
                Module.global_scope.pop()
            Module.is_add_module = False

            # Internal attributes
            module.parent = self
            module.is_subscope = is_subscope

            # Register module
            self.modules[name] = module
            if is_trainable:
                self.trainable_modules[name] = module

            return module

        else:
            raise TensorforceError.value(name='module specification', value=module)

    def get_variables(self, only_trainable=False):
        # only_trainable
        if not isinstance(only_trainable, bool):
            raise TensorforceError.type(name='get_variables', argument='only_trainable', value=only_trainable)

        if only_trainable:
            # Only trainable variables
            variables = list(self.trainable_variables.values())
            for module in self.trainable_modules.values():
                variables.extend(module.get_variables(only_trainable=only_trainable))

        else:
            # All variables
            variables = list(self.variables.values())
            for module in self.modules.values():
                variables.extend(module.get_variables(only_trainable=only_trainable))

        return variables
