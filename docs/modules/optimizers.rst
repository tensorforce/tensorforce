Optimizers
==========

Default optimizer: ``UpdateModifierWrapper``, so e.g. ``optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=5,  clipping_threshold=1e-2)``


.. autoclass:: tensorforce.core.optimizers.ClippingStep

.. autoclass:: tensorforce.core.optimizers.Evolutionary

.. autoclass:: tensorforce.core.optimizers.GlobalOptimizer

.. autoclass:: tensorforce.core.optimizers.MultiStep

.. autoclass:: tensorforce.core.optimizers.NaturalGradient

.. autoclass:: tensorforce.core.optimizers.OptimizingStep

.. autoclass:: tensorforce.core.optimizers.Plus

.. autoclass:: tensorforce.core.optimizers.SubsamplingStep

.. autoclass:: tensorforce.core.optimizers.Synchronization

.. autoclass:: tensorforce.core.optimizers.TFOptimizer

.. automethod:: tensorforce.core.optimizers.UpdateModifierWrapper
