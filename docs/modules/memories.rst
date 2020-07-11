Memories
========

Default memory: ``Replay`` with default argument ``capacity``, so an ``int`` is a short-form specification of a replay memory with corresponding capacity:

.. code-block:: python

    Agent.create(
        ...
        memory=10000,
        ...
    )


.. autoclass:: tensorforce.core.memories.Replay

.. autoclass:: tensorforce.core.memories.Recent
