Policies
========

Default policy: ``ParametrizedDistributions`` with default argument ``network`` (with default argument ``layers``), so a ``list`` is a short-form specification of a sequential layer-stack network architecture:

.. code-block:: python

    Agent.create(
        ...
        policy=[
            dict(type='dense', size=64, activation='tanh'),
            dict(type='dense', size=64, activation='tanh')
        ],
        ...
    )

See the `networks documentation <networks.html>`_ for more information about how to specify a network.

Example of a full policy specification with customized distribution and decaying temperature:

.. code-block:: python

    Agent.create(
        ...
        policy=dict(
            network=[
                dict(type='dense', size=64, activation='tanh'),
                dict(type='dense', size=64, activation='tanh')
            ],
            distributions=dict(
                float=dict(type='gaussian', global_stddev=True),
                bounded_action=dict(type='beta')
            ),
            temperature=dict(
                type='decaying', decay='exponential', unit='episodes',
                num_steps=100, initial_value=0.01, decay_rate=0.5
            )
        )
        ...
    )


.. autoclass:: tensorforce.core.policies.ParametrizedDistributions
