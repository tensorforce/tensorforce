General agent interface
=======================

Initialization and termination
------------------------------

.. automethod:: tensorforce.agents.TensorforceAgent.create
.. automethod:: tensorforce.agents.TensorforceAgent.reset
.. automethod:: tensorforce.agents.TensorforceAgent.close

Main reinforcement learning interface
-------------------------------------

.. automethod:: tensorforce.agents.TensorforceAgent.act
.. automethod:: tensorforce.agents.TensorforceAgent.observe

Required for independent act at episode start
---------------------------------------------

.. automethod:: tensorforce.agents.TensorforceAgent.initial_internals

Loading and saving
------------------

.. automethod:: tensorforce.agents.TensorforceAgent.load
.. automethod:: tensorforce.agents.TensorforceAgent.save

Advanced functions for specialized use cases
--------------------------------------------

.. automethod:: tensorforce.agents.TensorforceAgent.experience
.. automethod:: tensorforce.agents.TensorforceAgent.update
.. automethod:: tensorforce.agents.TensorforceAgent.pretrain

Get and assign variables
------------------------

.. automethod:: tensorforce.agents.TensorforceAgent.get_variables
.. automethod:: tensorforce.agents.TensorforceAgent.get_variable
.. automethod:: tensorforce.agents.TensorforceAgent.assign_variable
