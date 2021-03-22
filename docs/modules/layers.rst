Layers
======

See the `networks documentation <networks.html>`_ for more information about how to specify networks.

Default layer: ``Function`` with default argument ``function``, so a ``lambda`` function is a short-form specification of a simple transformation layer:

.. code-block:: python

    Agent.create(
        ...
        policy=dict(network=[
            (lambda x: tf.clip_by_value(x, -1.0, 1.0)),
            ...
        ]),
        ...
    )


Dense layers
------------

.. autoclass:: tensorforce.core.layers.Dense

.. autoclass:: tensorforce.core.layers.Linear


Convolutional layers
--------------------

.. autoclass:: tensorforce.core.layers.Conv1d

.. autoclass:: tensorforce.core.layers.Conv2d

.. autoclass:: tensorforce.core.layers.Conv1dTranspose

.. autoclass:: tensorforce.core.layers.Conv2dTranspose


Embedding layers
----------------

.. autoclass:: tensorforce.core.layers.Embedding


Recurrent layers (unrolled over timesteps)
------------------------------------------

.. autoclass:: tensorforce.core.layers.Rnn

.. autoclass:: tensorforce.core.layers.Lstm

.. autoclass:: tensorforce.core.layers.Gru


Input recurrent layers (unrolled over sequence input)
-----------------------------------------------------

.. autoclass:: tensorforce.core.layers.InputRnn

.. autoclass:: tensorforce.core.layers.InputLstm

.. autoclass:: tensorforce.core.layers.InputGru


Pooling layers
--------------

.. autoclass:: tensorforce.core.layers.Flatten

.. autoclass:: tensorforce.core.layers.Pooling

.. autoclass:: tensorforce.core.layers.Pool1d

.. autoclass:: tensorforce.core.layers.Pool2d


Normalization layers
--------------------

.. autoclass:: tensorforce.core.layers.LinearNormalization

.. autoclass:: tensorforce.core.layers.ExponentialNormalization

.. autoclass:: tensorforce.core.layers.InstanceNormalization

.. autoclass:: tensorforce.core.layers.BatchNormalization


Misc layers
-----------

.. autoclass:: tensorforce.core.layers.Reshape

.. autoclass:: tensorforce.core.layers.Activation

.. autoclass:: tensorforce.core.layers.Dropout

.. autoclass:: tensorforce.core.layers.Clipping

.. autoclass:: tensorforce.core.layers.Image

.. autoclass:: tensorforce.core.layers.Deltafier

.. autoclass:: tensorforce.core.layers.Sequence


Special layers
--------------

.. autoclass:: tensorforce.core.layers.Function

.. autoclass:: tensorforce.core.layers.Register

.. autoclass:: tensorforce.core.layers.Retrieve

.. autoclass:: tensorforce.core.layers.Block

.. autoclass:: tensorforce.core.layers.Reuse


Keras layer
-----------

.. autoclass:: tensorforce.core.layers.KerasLayer
