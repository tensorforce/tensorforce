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

import functools
import importlib
import json
import os
import re

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
import tensorforce.core
from tensorforce.core import SignatureDict, TensorSpec, tf_util


def make_key(*, x):
    try:
        hash(x)
        if x is not None:
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


def tf_function(
    *, num_args, optional=0, overwrites_signature=False, is_loop_body=False, dict_interface=False
):

    def decorator(function):

        def decorated(self, *args, **kwargs):
            assert len(args) == 0 or len(kwargs) == 0
            assert len(args) == 0 or len(args) == num_args

            # Function name and qualname
            name = function.__name__
            qualname = function.__qualname__

            # Parameters-to-graph mapping
            if not hasattr(self, '_{name}_graphs'.format(name=name)):
                setattr(self, '_{name}_graphs'.format(name=name), dict())
                assert function.__qualname__.endswith('.' + name)
                setattr(self, '_{name}_qualname'.format(name=name), function.__qualname__)
            function_graphs = getattr(self, '_{name}_graphs'.format(name=name))
            qualname = getattr(self, '_{name}_qualname'.format(name=name))

            # Handle overwriting signature
            if overwrites_signature:
                setattr(self, '_{name}_overwritten'.format(name=name), overwrites_signature)
            overwritten = getattr(self, '_{name}_overwritten'.format(name=name), False)

            # Graph signature
            input_signature = self.input_signature(function=name)
            output_signature = self.output_signature(function=name)

            # Apply raw function if qualname mismatch, which indicates super() call
            if function.__qualname__ != qualname:
                if not overwritten:
                    assert num_args - optional <= input_signature.num_args() <= num_args
                return function(self, *args, **kwargs)

            # Check number of arguments
            assert num_args - optional <= input_signature.num_args() <= num_args

            # Graph arguments
            if len(kwargs) > 0:
                graph_args = input_signature.kwargs_to_args(
                    kwargs=kwargs, to_dict=dict_interface, outer_tuple=True
                )
            else:
                graph_args = args

            # Graph parameters
            params_kwargs = {
                key: arg for key, arg in kwargs.items() if key not in input_signature
            }
            graph_params = tuple(make_key(x=arg) for arg in params_kwargs.values())

            # Check whether output_signature is parametrized
            if not isinstance(output_signature, SignatureDict):
                output_signature = output_signature(**params_kwargs)

            # Function graph
            if str(graph_params) not in function_graphs:

                def function_graph(*args):
                    with self:
                        # TODO: tf.name_scope instead?
                        kwargs = input_signature.args_to_kwargs(args=args, from_dict=dict_interface)
                        args = function(self, **kwargs.to_kwargs(), **params_kwargs)
                        args = output_signature.kwargs_to_args(kwargs=args, to_dict=dict_interface)
                    return args

                function_graph.__name__ = name
                function_graph.__qualname__ = qualname

                function_graphs[str(graph_params)] = tf.function(
                    func=function_graph,
                    input_signature=input_signature.to_list(to_dict=dict_interface),
                    autograph=False
                    # experimental_implements=None, experimental_autograph_options=None,
                    # experimental_relax_shapes=False, experimental_compile=None
                )

            # Apply function graph
            output_args = function_graphs[str(graph_params)](*graph_args)
            if not is_loop_body:
                return output_signature.args_to_kwargs(
                    args=output_args, outer_tuple=True, from_dict=dict_interface
                )
            else:
                return output_args

        return decorated

    return decorator


class Module(tf.Module):
    """
    Base class for modules.

    Args:
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    _TF_MODULE_IGNORED_PROPERTIES = \
        tf.Module._TF_MODULE_IGNORED_PROPERTIES | {'_MODULE_STACK', 'parent'}

    # _MODULE_STACK  # Initialized as part of model.__init__()

    def __init__(self, *, device=None, l2_regularization=None, name=None):
        super().__init__(name=name)

        self.checkpoint = None
        self.is_trainable = None
        self.is_saved = None
        self.is_initialized = None

        assert len(Module._MODULE_STACK) >= 1
        if isinstance(Module._MODULE_STACK[-1], type):
            assert isinstance(self, Module._MODULE_STACK[-1])
        else:
            # Not always type, e.g. tf_optimizer uses functools.partial
            assert isinstance(Module._MODULE_STACK[-1], functools.partial) and \
                isinstance(self, Module._MODULE_STACK[-1].func)
        Module._MODULE_STACK[-1] = self
        if len(Module._MODULE_STACK) > 1:
            self.parent = Module._MODULE_STACK[-2]
        else:
            self.parent = None

        # Device
        if device is None:
            self.device = util.NullContext()
        else:
            self.device = tf.device(device_name=device)

        # L2 regularization
        if l2_regularization is None:
            self.l2_regularization = None
        else:
            self.l2_regularization = self.submodule(
                name='l2_regularization', module=l2_regularization,
                modules=tensorforce.core.parameter_modules, is_trainable=False, dtype='float',
                min_value=0.0
            )

    @property
    def root(self):
        return self.parent.root

    @property
    def config(self):
        return self.parent.config

    @property
    def full_name(self):
        return '{}/{}'.format(self.parent.full_name, self.name)

    @property
    def tensorforce_submodules(self):
        predicate = (lambda x: isinstance(x, Module))
        return list(self._flatten(recursive=True, predicate=predicate))

    @property
    def this_submodules(self):
        predicate = (lambda x: isinstance(x, tf.Module))
        return list(self._flatten(recursive=False, predicate=predicate))

    @property
    def this_trainable_variables(self):
        predicate = (lambda x: isinstance(x, tf.Variable) and getattr(x, 'trainable', False))
        return list(self._flatten(recursive=False, predicate=predicate))

    # @property
    # def trainable_variables(self):
    #     predicate = (lambda x: isinstance(x, tf.Variable) and getattr(x, 'trainable', False))
    #     variables = list(self._flatten(recursive=False, predicate=predicate))
    #     for module in self.this_submodules:
    #         # if not isinstance(module, Module) or module.is_trainable:
    #         variables.extend(module.trainable_variables)
    #     return variables

    @property
    def saved_variables(self):
        predicate = (lambda x: isinstance(x, tf.Variable) and getattr(x, 'is_saved', True))
        variables = list(self._flatten(recursive=False, predicate=predicate))
        for module in self.this_submodules:
            if not isinstance(module, Module):
                variables.extend(module.variables)
            elif module.is_saved:
                variables.extend(module.saved_variables)
        return variables

    def __enter__(self):
        Module._MODULE_STACK.append(self)
        self.device.__enter__()
        assert isinstance(self.is_initialized, bool)
        if self.is_initialized:
            self.name_scope.__enter__()
        else:
            self._proper_name_scope = tf.name_scope(name=self.name)
            self._proper_name_scope.__enter__()
        return self

    def __exit__(self, etype, exception, traceback):
        if self.is_initialized:
            self.name_scope.__exit__(etype, exception, traceback)
        else:
            self._proper_name_scope.__exit__(etype, exception, traceback)
        self.device.__exit__(etype, exception, traceback)
        popped = Module._MODULE_STACK.pop()
        assert popped is self

    def initialize(self):
        self.summary_steps = dict()
        assert self.is_initialized is False
        for module in self.this_submodules:
            if isinstance(module, Module):
                assert module.is_initialized is None
                module.is_initialized = False
                with module:
                    module.initialize()
                assert module.is_initialized is False
                module.is_initialized = True

    def save(self, *, directory, filename=None):
        if filename is None:
            filename = self.full_name.replace('/', '.')
        if self.checkpoint is None:
            self.checkpoint = tf.train.Checkpoint(**{self.name: self})
        return self.checkpoint.write(file_prefix=os.path.join(directory, filename))

    def restore(self, *, directory, filename=None):
        if filename is None:
            filename = self.full_name.replace('/', '.')
        if self.checkpoint is None:
            self.checkpoint = tf.train.Checkpoint(**{self.name: self})
        try:
            self.checkpoint.restore(save_path=os.path.join(directory, filename)).expect_partial()
        except AssertionError as exc:
            if len(exc.args) != 1 or not re.match(
                pattern=r"Some Python objects were not bound to checkpointed values, likely due to "
                        r"changes in the Python program: \[<tf\.Variable 'save_counter:0' "
                        r"shape=\(\) dtype=int64, numpy=[0-9]*>(, <tf\.Variable 'save_counter:0' "
                        r"shape=\(\) dtype=int64, numpy=[0-9]*>)*\]",
                string=exc.args[0]
            ):
                raise exc

    def input_signature(self, *, function):
        if function == 'regularize':
            return SignatureDict()

        else:
            raise NotImplementedError

    def output_signature(self, *, function):
        if function == 'regularize':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=()).signature(batched=False)
            )

        else:
            raise NotImplementedError

    @tf_function(num_args=0)
    def regularize(self):
        zero = tf_util.constant(value=0.0, dtype='float')

        module = self
        while module.l2_regularization is None:
            module = module.parent

        if len(self.this_trainable_variables) == 0 or \
                module.l2_regularization.is_constant(value=0.0):
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

    def submodule(
        self, *, name, module=None, modules=None, default_module=None, is_trainable=True,
        is_saved=True, **kwargs
    ):
        assert self.is_initialized is None

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
        Module._MODULE_STACK.append(module_cls)
        module = module_cls(*args, **kwargs)
        popped = Module._MODULE_STACK.pop()
        assert popped is module

        assert module.is_trainable is None
        module.is_trainable = is_trainable
        assert module.is_saved is None
        module.is_saved = is_saved

        return module

    def variable(
        self, *, name, spec, initializer, is_trainable, is_saved, initialization_scale=None
    ):
        assert self.is_initialized is False
        # name
        if not isinstance(name, str):
            raise TensorforceError.type(name='variable', argument='name', dtype=type(name))
        # spec
        if not isinstance(spec, TensorSpec):
            raise TensorforceError.dtype(name='variable', argument='spec', dtype=type(spec))
        if spec.is_underspecified():
            raise TensorforceError.value(
                name='variable', argument='spec', value=spec, hint='underspecified'
            )
        # initializer
        initializer_names = (
            'constant', 'normal', 'normal-relu', 'ones', 'orthogonal', 'orthogonal-relu', 'zeros'
        )
        if not isinstance(initializer, (spec.py_type(), np.ndarray, tf.Tensor)) and \
                initializer not in initializer_names:
            raise TensorforceError.value(name='variable', argument='initializer', value=initializer)
        elif isinstance(initializer, np.ndarray) and initializer.dtype != spec.np_type():
            raise TensorforceError.type(
                name='variable', argument='initializer', dtype=initializer.dtype
            )
        elif isinstance(initializer, tf.Tensor) and tf_util.dtype(x=initializer) != spec.tf_type():
            raise TensorforceError.type(
                name='variable', argument='initializer', dtype=tf_util.dtype(x=initializer)
            )
        # initialization_scale
        if initialization_scale is not None:
            if isinstance(initializer, (spec.py_type(), np.ndarray, tf.Tensor)) or \
                    initializer not in ('constant', 'orthogonal', 'orthogonal-relu'):
                raise TensorforceError.invalid(
                    name='variable', argument='initialization_scale',
                    condition='initializer not orthogonal'
                )
            elif not isinstance(initialization_scale, spec.py_type()):
                raise TensorforceError.type(
                    name='variable', argument='initialization_scale',
                    dtype=type(initialization_scale), hint='!= float'
                )
        # is_trainable
        if not isinstance(is_trainable, bool):
            raise TensorforceError.type(
                name='variable', argument='is_trainable', dtype=type(is_trainable)
            )
        elif is_trainable and spec.type != 'float':
            raise TensorforceError.value(
                name='variable', argument='is_trainable', value=is_trainable,
                condition='spec.type != float'
            )
        # is_saved
        if not isinstance(is_saved, bool):
            raise TensorforceError.type(name='variable', argument='is_saved', dtype=type(is_saved))

        # Variable initializer
        if isinstance(initializer, spec.py_type()):
            initializer = tf_util.constant(value=initializer, dtype=spec.type, shape=spec.shape)
        elif isinstance(initializer, np.ndarray):
            if initializer.shape != spec.shape:
                raise TensorforceError.mismatch(
                    name='Module.variable', value1='shape', value2='initializer'
                )
            initializer = tf_util.constant(value=initializer, dtype=spec.type)
        elif isinstance(initializer, tf.Tensor):
            if tf_util.shape(x=initializer) != spec.shape:
                raise TensorforceError.mismatch(
                    name='Module.variable', value1='shape', value2='initializer'
                )
            initializer = initializer
        elif not isinstance(initializer, str):
            raise TensorforceError("Invalid variable initializer: {}".format(initializer))
        elif initializer.startswith('normal'):
            if spec.type != 'float':
                raise TensorforceError(
                    message="Invalid variable initializer value for non-float variable: {}.".format(
                        initializer
                    )
                )
            if initializer.endswith('-relu'):
                stddev = min(0.1, np.sqrt(2.0 / util.product(xs=spec.shape[:-1])))
            else:
                stddev = min(0.1, np.sqrt(2.0 / (util.product(xs=spec.shape[:-1]) + spec.shape[-1])))
            initializer = tf.random.normal(shape=spec.shape, stddev=stddev, dtype=spec.tf_type())
        elif initializer.startswith('orthogonal'):
            if spec.type != 'float':
                raise TensorforceError(
                    message="Invalid variable initializer value for non-float variable: {}.".format(
                        initializer
                    )
                )
            if spec.rank < 2:
                raise TensorforceError(
                    message="Invalid variable initializer value for 0/1-rank variable: {}.".format(
                        initializer
                    )
                )
            normal = np.random.normal(size=(util.product(xs=spec.shape[:-1]), spec.shape[-1]))
            u, _, v = np.linalg.svd(a=normal, full_matrices=False)
            orthogonal = u if u.shape[1] == spec.shape[-1] else v
            if initializer.endswith('-relu'):
                orthogonal = orthogonal * np.sqrt(2.0)
            if initialization_scale is not None and initialization_scale != 1.0:
                if initialization_scale <= 0.0:
                    raise TensorforceError.value(
                        name='variable', argument='initialization_scale',
                        value=initialization_scale, hint='<= 0.0'
                    )
                orthogonal = orthogonal * initialization_scale
            initializer = tf_util.constant(value=orthogonal.reshape(spec.shape), dtype=spec.type)
        elif initializer == 'zeros':
            initializer = tf_util.zeros(shape=spec.shape, dtype=spec.type)
        elif initializer == 'ones':
            initializer = tf_util.ones(shape=spec.shape, dtype=spec.type)
        elif initializer == 'constant':
            initializer = tf.fill(
                dims=spec.shape, value=tf_util.constant(value=initialization_scale, dtype=spec.type)
            )

        # Variable
        variable = tf.Variable(
            initial_value=initializer, trainable=is_trainable, validate_shape=True, name=name,
            dtype=spec.tf_type(), shape=spec.shape
        )
        variable.is_saved = is_saved

        return variable

    def register_summary(self, *, label, name):
        # label
        if not isinstance(label, str):
            raise TensorforceError.type(name='Module.summary', argument='label', dtype=type(label))
        # name
        if not isinstance(name, (str, list, tuple)):
            raise TensorforceError.type(name='Module.summary', argument='name', dtype=type(name))
        if len(name) == 0:
            raise TensorforceError.required(name='Module.summary', argument='name')
        if not isinstance(name, str):
            name = name[0]
        if name in self.summary_steps:
            raise TensorforceError.value(
                name='Module.summary', argument='name', hint='already exists'
            )

        if self.root.summaries == 'all' or label in self.root.summaries:
            self.summary_steps[name] = self.variable(
                name=(name + '-summary'), spec=TensorSpec(type='int'), initializer=-1,
                is_trainable=False, is_saved=False
            )

    def summary(self, *, label, name, data, step):
        # label
        if not isinstance(label, str):
            raise TensorforceError.type(name='Module.summary', argument='label', dtype=type(label))
        # name
        if not isinstance(name, (str, list, tuple)):
            raise TensorforceError.type(name='Module.summary', argument='name', dtype=type(name))
        if len(name) == 0:
            raise TensorforceError.required(name='Module.summary', argument='name')
        if isinstance(name, str):
            name = (name,)
        # data
        if not tf_util.is_tensor(x=data) and not callable(data):
            raise TensorforceError.type(name='Module.summary', argument='data', dtype=type(data))
        # step
        if step not in self.root.units:
            raise TensorforceError.value(name='Module.summary', argument='step', value=step)

        if self.root.summaries == 'all' or label in self.root.summaries:
            if name[0] not in self.summary_steps:
                raise TensorforceError.value(
                    name='Module.summary', argument='name', value=name, hint='is not registered'
                )

            unit = self.root.units[step]

            def write_summary():
                if callable(data):
                    d = data()
                else:
                    d = data
                if tf_util.is_tensor(x=d):
                    d = (d,)
                dependencies = list()
                with self.root.summarizer.as_default():
                    for n, d in zip(name, d):
                        dependencies.append(tf.summary.scalar(name=n, data=d, step=unit))
                previous = self.summary_steps[name[0]]
                dependencies.append(previous.assign(value=unit, read_value=False))
                return tf.group(*dependencies)

            pred = unit > self.summary_steps[name[0]]
            return [tf.cond(pred=pred, true_fn=write_summary, false_fn=tf.no_op)]

        else:
            return list()
