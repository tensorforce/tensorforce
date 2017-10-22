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
import logging
import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError, Configuration


epsilon = 1e-6


log_levels = dict(
    info=logging.INFO,
    debug=logging.DEBUG,
    critical=logging.CRITICAL,
    warning=logging.WARNING,
    fatal=logging.FATAL
)


def prod(xs):
    """Computes the product along the elements in an iterable. Returns 1 for empty
        iterable.
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


def cumulative_discount(values, terminals, discount, cumulative_start=0.0):
    """
    Compute cumulative discounts.
    Args:
        values: Values to discount
        terminals: Booleans indicating terminal states
        discount: Discount factor
        cumulative_start: Float or ndarray, estimated reward for state t + 1. Default 0.0

    Returns:
        dicounted_values: The cumulative discounted rewards.
    """
    if discount == 0.0:
        return np.asarray(values)

    # cumulative start can either be a number or ndarray
    if type(cumulative_start) is np.ndarray:
        discounted_values = np.zeros((len(values),) + (cumulative_start.shape))
    else:
        discounted_values = np.zeros(len(values))

    cumulative = cumulative_start
    for n, (value, terminal) in reversed(list(enumerate(zip(values, terminals)))):
        if terminal:
            cumulative = np.zeros_like(cumulative_start, dtype=np.float32)
        cumulative = value + cumulative * discount
        discounted_values[n] = cumulative

    return discounted_values


def np_dtype(dtype):
    """Translates dtype specifications in configurations to numpy data types.
    Args:
        dtype: String describing a numerical type (e.g. 'float') or numerical type primitive.

    Returns: Numpy data type

    """
    if dtype == 'float' or dtype == float:
        return np.float32
    elif dtype == 'int' or dtype == int:
        return np.int32
    elif dtype == 'bool' or dtype == bool:
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
    if dtype == 'float' or dtype == float or dtype == np.float32:
        return tf.float32
    elif dtype == 'int' or dtype == int or dtype == np.int32:
        return tf.int32
    else:
        raise TensorForceError("Error: Type conversion from type {} not supported.".format(str(dtype)))


def get_function(fct, predefined=None):
    """
    Turn a function specification from a configuration to a function, e.g.
    by importing it from the respective module.
    Args:
        fct:
        predefined:

    Returns:

    """
    if predefined is not None and fct in predefined:
        return predefined[fct]
    elif isinstance(fct, str):
        module_name, function_name = fct.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    elif callable(fct):
        return fct
    else:
        raise TensorForceError("Argument {} cannot be turned into a function.".format(fct))


def get_object(obj, predefined=None, kwargs=None):
    """
    Utility method to map Configuration objects to their contents, e.g. optimisers, baselines
    to the respective classes.

    Args:
        obj: Configuration object or dict
        predefined: Default value
        kwargs: Parameters for the configuration

    Returns: The retrieved object

    """
    if isinstance(obj, Configuration):
        fct = obj.type
        full_kwargs = {key: value for key, value in obj if key != 'type'}
    elif isinstance(obj, dict):
        fct = obj['type']
        full_kwargs = {key: value for key, value in obj.items() if key != 'type'}
    else:
        fct = obj
        full_kwargs = dict()
    obj = get_function(fct=fct, predefined=predefined)
    if kwargs is not None:
        full_kwargs.update(kwargs)
    return obj(**full_kwargs)


class SummarySessionWrapper(object):
    def __init__(self, model, fetches, feed_dict):
        """
        A wrapper around a standard session.run() call that checks to see if the model should write summaries.  If so,
        it will write the summaries transparently.  Call the object's .run() method just like you would session.run(),
        and it returns fetches without the summaries.

        :param model: TensorForce model
        :param fetches:  What to fetch, just as you would with tf.session.run()
        :param feed_dict:   What to feed, just as you would with tf.session.run()
        """
        self._model = model
        self._session = model.session
        self._should_write_summaries = model.should_write_summaries()
        self._fetches = fetches
        if self._should_write_summaries:
            self._fetches.append(model.tf_summaries)
        self._feed_dict = feed_dict
        self._returns = None
        self._have_executed = False

    def __enter__(self):
        return self

    def _raise_duplicate_key_error(self, key):
        raise TensorForceError('Do not pass {} kwarg into run() for {}.  It is passed in constructor.'.format(
            self.__class__.__name__, key))

    def run(self, **kwargs):
        # check to make sure we don't accidentally pass in an altered fetches or feed dict
        for k in ('fetches', 'feed_dict'):
            if k in kwargs:
                self._raise_duplicate_key_error(k)
        self._returns = self._session.run(fetches=self._fetches, feed_dict=self._feed_dict, **kwargs)
        self._have_executed = True
        if self._should_write_summaries:
            # don't return the tf summaries we appended in __init__
            return self._returns[:-1]
        else:
            return self._returns

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is None:
            if not self._have_executed:
                raise TensorForceError('{} exiting without run() being called.'.format(self.__class__.__name__))
            if self._should_write_summaries:
                self._model.write_summaries(self._returns[-1])
                self._model.last_summary_step = self._model.timestep
        else:
            return 0
