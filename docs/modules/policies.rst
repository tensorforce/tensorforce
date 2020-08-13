Policies
========

Default policy: depends on agent configuration, but always with default argument ``network`` (with default argument ``layers``), so a ``list`` is a short-form specification of a sequential layer-stack network architecture:

.. code-block:: python

    Agent.create(
        ...
        policy=[
            dict(type='dense', size=64, activation='tanh'),
            dict(type='dense', size=64, activation='tanh')
        ],
        ...
    )

Or simply:

.. code-block:: python

    Agent.create(
        ...
        policy=dict(network='auto'),
        ...
    )

See the `networks documentation <networks.html>`_ for more information about how to specify a network.

Example of a full parametrized-distributions policy specification with customized distribution and decaying temperature:

.. code-block:: python

    Agent.create(
        ...
        policy=dict(
            type='parametrized_distributions',
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


.. autoclass:: tensorforce.core.policies.ParametrizedActionValue

.. autoclass:: tensorforce.core.policies.ParametrizedDistributions

.. autoclass:: tensorforce.core.policies.ParametrizedStateActionValue

.. autoclass:: tensorforce.core.policies.ParametrizedStateValue
