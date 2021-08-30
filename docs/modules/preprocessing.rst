Preprocessing
=============

Example of how to specify state and reward preprocessing:

.. code-block:: python

    Agent.create(
        ...
        reward_estimation=dict(
            ...
            reward_processing=dict(type='clipping', lower=-1.0, upper=1.0)
        ),
        state_preprocessing=[
            dict(type='image', height=4, width=4, grayscale=True),
            dict(type='exponential_normalization', decay=0.999)
        ],
        ...
    )


.. autoclass:: tensorforce.core.layers.Clipping
    :noindex:

.. autoclass:: tensorforce.core.layers.Image
    :noindex:

.. autoclass:: tensorforce.core.layers.LinearNormalization
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
