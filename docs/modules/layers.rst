Layers
======

Default layer: ``Function`` with default argument ``function``, so e.g. ``[..., (lambda x: x + 1.0), ...]``


Convolutional layers
--------------------

.. autoclass:: tensorforce.core.layers.Conv1d

.. autoclass:: tensorforce.core.layers.Conv2d


Dense layers
------------

.. autoclass:: tensorforce.core.layers.Dense

.. autoclass:: tensorforce.core.layers.Linear


Embedding layers
----------------

.. autoclass:: tensorforce.core.layers.Embedding


Recurrent layers
----------------

.. autoclass:: tensorforce.core.layers.Gru

.. autoclass:: tensorforce.core.layers.Lstm

.. autoclass:: tensorforce.core.layers.Rnn


Input recurrent layers
---------------------------

.. autoclass:: tensorforce.core.layers.InputGru

.. autoclass:: tensorforce.core.layers.InputLstm

.. autoclass:: tensorforce.core.layers.InputRnn


Pooling layers
--------------

.. autoclass:: tensorforce.core.layers.Flatten

.. autoclass:: tensorforce.core.layers.Pooling

.. autoclass:: tensorforce.core.layers.Pool1d

.. autoclass:: tensorforce.core.layers.Pool2d


Normalization layers
--------------------

.. autoclass:: tensorforce.core.layers.ExponentialNormalization

.. autoclass:: tensorforce.core.layers.InstanceNormalization


Misc layers
-----------

.. autoclass:: tensorforce.core.layers.Activation

.. autoclass:: tensorforce.core.layers.Clipping

.. autoclass:: tensorforce.core.layers.Deltafier

.. autoclass:: tensorforce.core.layers.Dropout

.. autoclass:: tensorforce.core.layers.Image

.. autoclass:: tensorforce.core.layers.Reshape

.. autoclass:: tensorforce.core.layers.Sequence


Special layers
--------------

.. autoclass:: tensorforce.core.layers.Block

.. autoclass:: tensorforce.core.layers.Function

.. autoclass:: tensorforce.core.layers.Keras

.. autoclass:: tensorforce.core.layers.Register

.. autoclass:: tensorforce.core.layers.Retrieve

.. autoclass:: tensorforce.core.layers.Reuse
