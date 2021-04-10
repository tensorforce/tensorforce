General agent interface
=======================

Initialization and termination
------------------------------

.. automethod:: tensorforce.agents.TensorforceAgent.create
.. automethod:: tensorforce.agents.TensorforceAgent.reset
.. automethod:: tensorforce.agents.TensorforceAgent.close

Reinforcement learning interface
--------------------------------

.. automethod:: tensorforce.agents.TensorforceAgent.act
.. automethod:: tensorforce.agents.TensorforceAgent.observe

Get initial internals (for independent-act)
-------------------------------------------

.. automethod:: tensorforce.agents.TensorforceAgent.initial_internals

Experience - update interface
-----------------------------

.. automethod:: tensorforce.agents.TensorforceAgent.experience
.. automethod:: tensorforce.agents.TensorforceAgent.update

Pretraining
-----------

.. automethod:: tensorforce.agents.TensorforceAgent.pretrain

Loading and saving
------------------

.. automethod:: tensorforce.agents.TensorforceAgent.load
.. automethod:: tensorforce.agents.TensorforceAgent.save

Tensor value tracking
---------------------

.. automethod:: tensorforce.agents.TensorforceAgent.tracked_tensors
