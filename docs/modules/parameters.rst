Parameters
==========

Tensorforce distinguishes between agent/module arguments (primitive types: bool/int/float) which either specify part of the TensorFlow model architecture, like the layer size, or a value within the architecture, like the learning rate. Whereas the former are statically defined as part of the agent initialization, the latter can be dynamically adjusted afterwards. These dynamic hyperparameter are indicated by ``parameter`` as part of their argument type specification in the documentation, and can alternatively be assigned a parameter module instead of a constant value, for instance, to specify a decaying learning rate.

Default parameter: ``Constant``, so a ``bool``/``int``/``float`` value is a short-form specification of a constant (dynamic) parameter:

.. code-block:: python

    Agent.create(
        ...
        exploration=0.1,
        ...
    )

Example of how to specify an exponentially decaying learning rate:

.. code-block:: python

    Agent.create(
        ...
        optimizer=dict(optimizer='adam', learning_rate=dict(
            type='exponential', unit='timesteps', num_steps=1000,
            initial_value=0.01, decay_rate=0.5
        )),
        ...
    )

Example of how to specify a linearly increasing reward horizon:

.. code-block:: python

    Agent.create(
        ...
        reward_estimation=dict(horizon=dict(
            type='linear', unit='episodes', num_steps=1000,
            initial_value=10, final_value=50
        )),
        ...
    )


.. autoclass:: tensorforce.core.parameters.Constant

.. autoclass:: tensorforce.core.parameters.Linear

.. autoclass:: tensorforce.core.parameters.PiecewiseConstant

.. autoclass:: tensorforce.core.parameters.Exponential

.. autoclass:: tensorforce.core.parameters.Decaying

.. autoclass:: tensorforce.core.parameters.OrnsteinUhlenbeck

.. autoclass:: tensorforce.core.parameters.Random
