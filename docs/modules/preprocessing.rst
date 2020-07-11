Preprocessing
=============

Example of how to specify state and reward preprocessing:

.. code-block:: python

    Agent.create(
        ...
        preprocessing=dict(
            state=[
                dict(type='image', height=32, width=32, grayscale=True),
                dict(type='exponential_normalization')
            ],
            reward=dict(type='clipping', lower=-1.0, upper=1.0)
        ),
        ...
    )


.. autoclass:: tensorforce.core.layers.Clipping
    :noindex:

.. autoclass:: tensorforce.core.layers.Image
    :noindex:

.. autoclass:: tensorforce.core.layers.ExponentialNormalization
    :noindex:

.. autoclass:: tensorforce.core.layers.InstanceNormalization
    :noindex:

.. autoclass:: tensorforce.core.layers.Deltafier
    :noindex:

.. autoclass:: tensorforce.core.layers.Sequence
    :noindex:

.. autoclass:: tensorforce.core.layers.Activation
    :noindex:

.. autoclass:: tensorforce.core.layers.Dropout
    :noindex:
