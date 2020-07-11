Distributions
=============

Distributions are customized via the ``distributions`` argument of ``policy``, for instance:

.. code-block:: python

    Agent.create(
        ...
        policy=dict(distributions=dict(
            float=dict(type='gaussian', global_stddev=True),
            bounded_action=dict(type='beta')
        ))
        ...
    )

See the `policies documentation <policies.html>`_ for more information about how to specify a policy.


.. autoclass:: tensorforce.core.distributions.Categorical

.. autoclass:: tensorforce.core.distributions.Gaussian

.. autoclass:: tensorforce.core.distributions.Bernoulli

.. autoclass:: tensorforce.core.distributions.Beta
