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

import importlib
import json
import logging
import os
import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError


epsilon = 1e-6


log_levels = dict(
    info=logging.INFO,
    debug=logging.DEBUG,
    critical=logging.CRITICAL,
    warning=logging.WARNING,
    fatal=logging.FATAL
)


def prod(xs):
    """Computes the product along the elements in an iterable. Returns 1 for empty iterable.

    Args:
        xs: Iterable containing numbers.

    Returns: Product along iterable.

    """
    p = 1
    for x in xs:
        p *= x
    return p


def rank(x):
    return x.get_shape().ndims


def shape(x, unknown=-1):
    return tuple(unknown if dims is None else dims for dims in x.get_shape().as_list())


def np_dtype(dtype):
    """Translates dtype specifications in configurations to numpy data types.
    Args:
        dtype: String describing a numerical type (e.g. 'float') or numerical type primitive.

    Returns: Numpy data type

    """
    if dtype == 'float' or dtype == float or dtype == np.float32 or dtype == tf.float32:
        return np.float32
    elif dtype == np.float64 or dtype == tf.float64:
        return np.float64
    elif dtype == np.float16 or dtype == tf.float16:
        return np.float16
    elif dtype == 'int' or dtype == int or dtype == np.int32 or dtype == tf.int32:
        return np.int32
    elif dtype == np.int64 or dtype == tf.int64:
        return np.int64
    elif dtype == np.int16 or dtype == tf.int16:
        return np.int16
    elif dtype == 'bool' or dtype == bool or dtype == np.bool_ or dtype == tf.bool:
        return np.bool_
    else:
        raise TensorForceError("Error: Type conversion from type {} not supported.".format(str(dtype)))


def tf_dtype(dtype):
    """Translates dtype specifications in configurations to tensorflow data types.

       Args:
           dtype: String describing a numerical type (e.g. 'float'), numpy data type,
               or numerical type primitive.

       Returns: TensorFlow data type

    """
    # Defaults to 32
    if dtype == 'float' or dtype == float or dtype == np.float32 or dtype == tf.float32:
        return tf.float32
    elif dtype == np.float64 or dtype == tf.float64:
        return tf.float64
    elif dtype == np.float16 or dtype == tf.float16:
        return tf.float16
    elif dtype == 'int' or dtype == int or dtype == np.int32 or dtype == tf.int32:
        return tf.int32
    elif dtype == np.int64 or dtype == tf.int64:
        return tf.int64
    elif dtype == np.int16 or dtype == tf.int16:
        return tf.int16
    elif dtype == 'bool' or dtype == bool or dtype == np.bool_ or dtype == tf.bool:
        return tf.bool
    else:
        raise TensorForceError("Error: Type conversion from type {} not supported.".format(str(dtype)))


def map_tensors(fn, tensors, index=None):
    if tensors is None:
        return None
    elif isinstance(tensors, tuple):
        return tuple(map_tensors(fn=fn, tensors=tensor) for tensor in tensors)
    elif isinstance(tensors, list):
        return [map_tensors(fn=fn, tensors=tensor) for tensor in tensors]
    elif isinstance(tensors, dict) and index is None:
        return {key: map_tensors(fn=fn, tensors=tensor) for key, tensor in tensors.items()}
    elif isinstance(tensors, dict) and index is not None:
        return {key: map_tensors(fn=fn, tensors=tensor[index]) for key, tensor in tensors.items()}
    elif isinstance(tensors, set):
        return {map_tensors(fn=fn, tensors=tensor) for tensor in tensors}
    else:
        return fn(tensors)


def get_tensor_dependencies(tensor):
    """
    Utility method to get all dependencies (including placeholders) of a tensor (backwards through the graph).

    Args:
        tensor (tf.Tensor): The input tensor.

    Returns: Set of all dependencies (including needed placeholders) for the input tensor.
    """
    dependencies = set()
    dependencies.update(tensor.op.inputs)
    for sub_op in tensor.op.inputs:
        dependencies.update(get_tensor_dependencies(sub_op))
    return dependencies


def get_object(obj, predefined_objects=None, default_object=None, kwargs=None):
    """
    Utility method to map some kind of object specification to its content,
    e.g. optimizer or baseline specifications to the respective classes.

    Args:
        obj: A specification dict (value for key 'type' optionally specifies
                the object, options as follows), a module path (e.g.,
                my_module.MyClass), a key in predefined_objects, or a callable
                (e.g., the class type object).
        predefined_objects: Dict containing predefined set of objects,
                accessible via their key
        default_object: Default object is no other is specified
        kwargs: Arguments for object creation

    Returns: The retrieved object

    """
    args = ()
    kwargs = dict() if kwargs is None else kwargs

    if isinstance(obj, str) and os.path.isfile(obj):
        with open(obj, 'r') as fp:
            obj = json.load(fp=fp)
    if isinstance(obj, dict):
        kwargs.update(obj)
        obj = kwargs.pop('type', None)

    if predefined_objects is not None and obj in predefined_objects:
        obj = predefined_objects[obj]
    elif isinstance(obj, str):
        if obj.find('.') != -1:
            module_name, function_name = obj.rsplit('.', 1)
            module = importlib.import_module(module_name)
            obj = getattr(module, function_name)
        else:
            raise TensorForceError("Error: object {} not found in predefined objects: {}".format(
                obj,
                list(predefined_objects or ())
            ))
    elif callable(obj):
        pass
    elif default_object is not None:
        args = (obj,)
        obj = default_object
    else:
        # assumes the object is already instantiated
        return obj

    return obj(*args, **kwargs)


def prepare_kwargs(raw, string_parameter='name'):
    """
    Utility method to convert raw string/diction input into a dictionary to pass
    into a function.  Always returns a dictionary.

    Args:
        raw: string or dictionary, string is assumed to be the name of the activation
                activation function.  Dictionary will be passed through unchanged.

    Returns: kwargs dictionary for **kwargs

    """
    kwargs = dict()

    if isinstance(raw, dict):
        kwargs.update(raw)
    elif isinstance(raw, str):
        kwargs[string_parameter] = raw

    return kwargs


def strip_name_scope(name, base_scope):
    if name.startswith(base_scope):
        return name[len(base_scope):]
    else:
        return name


class SavableComponent(object):
    """
    Component that can save and restore its own state.
    """

    def register_saver_ops(self):
        """
        Registers the saver operations to the graph in context.
        """

        variables = self.get_savable_variables()
        if variables is None or len(variables) == 0:
            self._saver = None
            return

        base_scope = self._get_base_variable_scope()
        variables_map = {strip_name_scope(v.name, base_scope): v for v in variables}

        self._saver = tf.train.Saver(
            var_list=variables_map,
            reshape=False,
            sharded=False,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            saver_def=None,
            builder=None,
            defer_build=False,
            allow_empty=True,
            write_version=tf.train.SaverDef.V2,
            pad_step_number=False,
            save_relative_paths=True
        )

    def get_savable_variables(self):
        """
        Returns the list of all the variables this component is responsible to save and restore.

        Returns:
            The list of variables that will be saved or restored.
        """

        raise NotImplementedError()

    def save(self, sess, save_path, timestep=None):
        """
        Saves this component's managed variables.

        Args:
            sess: The session for which to save the managed variables.
            save_path: The path to save data to.
            timestep: Optional, the timestep to append to the file name.

        Returns:
            Checkpoint path where the model was saved.
        """

        if self._saver is None:
            raise TensorForceError("register_saver_ops should be called before save")
        return self._saver.save(
            sess=sess,
            save_path=save_path,
            global_step=timestep,
            write_meta_graph=False,
            write_state=True,  # Do we need this?
        )

    def restore(self, sess, save_path):
        """
        Restores the values of the managed variables from disk location.

        Args:
            sess: The session for which to save the managed variables.
            save_path: The path used to save the data to.
        """

        if self._saver is None:
            raise TensorForceError("register_saver_ops should be called before restore")
        self._saver.restore(sess=sess, save_path=save_path)

    def _get_base_variable_scope(self):
        """
        Returns the portion of the variable scope that is considered a base for this component. The variables will be
        saved with names relative to that scope.

        Returns:
            The name of the base variable scope, should always end with "/".
        """

        raise NotImplementedError()
