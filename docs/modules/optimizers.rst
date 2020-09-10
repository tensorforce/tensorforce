Optimizers
==========

Default optimizer: ``OptimizerWrapper`` which offers additional update modifier options, so instead of using ``TFOptimizer`` directly, a customized Adam optimizer can be specified via:

.. code-block:: python

    Agent.create(
        ...
        optimizer=dict(
            optimizer='adam', learning_rate=1e-3, clipping_threshold=1e-2,
            multi_step=10, subsampling_fraction=64, linesearch_iterations=5,
            doublecheck_update=True
        ),
        ...
    )


.. autoclass:: tensorforce.core.optimizers.OptimizerWrapper

.. autoclass:: tensorforce.core.optimizers.TFOptimizer

.. autoclass:: tensorforce.core.optimizers.NaturalGradient

.. autoclass:: tensorforce.core.optimizers.Evolutionary

.. autoclass:: tensorforce.core.optimizers.ClippingStep

.. autoclass:: tensorforce.core.optimizers.MultiStep

.. autoclass:: tensorforce.core.optimizers.DoublecheckStep

.. autoclass:: tensorforce.core.optimizers.LinesearchStep

.. autoclass:: tensorforce.core.optimizers.SubsamplingStep

.. autoclass:: tensorforce.core.optimizers.Synchronization

.. autoclass:: tensorforce.core.optimizers.Plus
