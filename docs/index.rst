*TensorForce - modular deep reinforcement learning in TensorFlow*
=================================================================================

TensorForce is an open source reinforcement learning library focused on
providing clear APIs, readability and modularisation to deploy
reinforcement learning solutions both in research and practice.
TensorForce is built on top on TensorFlow.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   agents_models.rst
   preprocessing.rst
   runner.rst

Quick start
-----------

For a quick start, you can run one of our example scripts using the
provided configurations, e.g. to run the TRPO agent on CartPole, execute
from the examples folder:

::

    python tensorforce/examples/openai_gym.py CartPole-v0 -a TRPOAgent -c tensorforce/examples/configs/trpo_agent.json -n tensorforce/examples/configs/trpo_network.json
    

More information
----------------

You can find more information at our `TensorForce GitHub repository <https://github.com/reinforceio/TensorForce>`__.

